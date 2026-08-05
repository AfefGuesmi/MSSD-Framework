# -*- coding: utf-8 -*-
"""
Data loader for pixel-level semantic segmentation of Sentinel-2 patches.

Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
"""

import logging
import os
import random
from os.path import dirname as up

import numpy as np
import torch
import torchvision.transforms.functional as F
from osgeo import gdal
from torch.utils.data import Dataset
from tqdm import tqdm

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Pixel-level class distribution (sum = 1.0)
CLASS_DISTR = torch.tensor([
    0.00452, 0.00203, 0.00254, 0.00168, 0.00766,
    0.15206, 0.20232, 0.35941, 0.00109, 0.20218,
    0.03226, 0.00693, 0.01322, 0.01158, 0.00052
])

# Sentinel-2 band means and stds (11 bands)
BANDS_MEAN = np.array([
    0.05197577, 0.04783991, 0.04056812, 0.03163572,
    0.02972606, 0.03457443, 0.03875053, 0.03436435,
    0.0392113, 0.02358126, 0.01588816
], dtype=np.float32)

BANDS_STD = np.array([
    0.04725893, 0.04743808, 0.04699043, 0.04967381,
    0.04946782, 0.06458357, 0.07594915, 0.07120246,
    0.08251058, 0.05111466, 0.03524419
], dtype=np.float32)

# Path to data directory (relative to this file)
DATASET_PATH = os.path.join(up(__file__), 'data')


class GenDEBRIS(Dataset):
    """
    PyTorch Dataset for MARIDA Sentinel-2 patches.

    Args:
        mode (str): 'train', 'val', or 'test'
        transform (callable, optional): Transform to apply to image and mask.
        standardization (callable, optional): Normalization using band stats.
        path (str): Root directory of the dataset.
        agg_to_water (bool): If True, merge classes 12-15 into Marine Water.
        rare_classes (list[int], optional): 0-indexed class IDs (in the same
            aggregated label space __getitem__ returns) used by both
            copy-paste augmentation and compute_sample_weights().
        copy_paste_prob (float): Probability, per __getitem__ call, of
            pasting rare-class pixels from a randomly chosen donor patch
            (one known to contain at least one rare class) onto the
            current patch at the same spatial positions. 0 disables it.
            Intended for the train split only -- leave at 0 for val/test
            so evaluation stays deterministic and unaugmented.
        spectral_jitter_prob (float): Probability, per __getitem__ call, of
            multiplying each band by an independent random factor close to
            1.0 (see spectral_jitter_strength), simulating sensor/
            atmospheric variation. 0 disables it. Train split only.
        spectral_jitter_strength (float): Each band's multiplicative jitter
            factor is drawn uniformly from
            [1 - spectral_jitter_strength, 1 + spectral_jitter_strength].
    """

    def __init__(self, mode='train', transform=None, standardization=None,
                 path=DATASET_PATH, agg_to_water=True, rare_classes=None,
                 copy_paste_prob=0.0, spectral_jitter_prob=0.0,
                 spectral_jitter_strength=0.05):
        super().__init__()

        split_file = os.path.join(path, 'splits', f'{mode}_X.txt')
        self.rois = np.genfromtxt(split_file, dtype='str')

        self.images = []   # list of image arrays (C, H, W)
        self.masks = []    # list of mask arrays (H, W)

        for roi in tqdm(self.rois, desc=f'Load {mode} set to memory'):
            # Build file paths
            folder = '_'.join(['S2'] + roi.split('_')[:-1])
            name = '_'.join(['S2'] + roi.split('_'))
            img_path = os.path.join(path, 'patches', folder, f'{name}.tif')
            mask_path = os.path.join(path, 'patches', folder, f'{name}_cl.tif')

            # Load mask
            ds = gdal.Open(mask_path)
            mask = ds.ReadAsArray().astype(np.int64)
            ds = None

            # Optionally aggregate water classes
            if agg_to_water:
                # Mixed Water(15), Wakes(14), Cloud Shadows(13), Waves(12) -> Marine Water(7)
                mask[mask == 15] = 7
                mask[mask == 14] = 7
                mask[mask == 13] = 7
                mask[mask == 12] = 7

            # Shift classes from 1..15 to 0..14
            mask = mask - 1
            self.masks.append(mask)

            # Load image
            ds = gdal.Open(img_path)
            img = ds.ReadAsArray()
            ds = None
            self.images.append(img)

        # Pre‑compute nan imputation array
        sample_img = self.images[0]
        self.impute_nan = np.tile(BANDS_MEAN, (sample_img.shape[1], sample_img.shape[2], 1))

        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        self.path = path
        self.agg_to_water = agg_to_water

        self.rare_classes = list(rare_classes) if rare_classes else []
        self.copy_paste_prob = copy_paste_prob
        self.spectral_jitter_prob = spectral_jitter_prob
        self.spectral_jitter_strength = spectral_jitter_strength

        self._rare_class_patch_indices = []
        if self.copy_paste_prob > 0:
            if not self.rare_classes:
                logging.warning(
                    "GenDEBRIS(%s): copy_paste_prob=%.2f but rare_classes is empty -- "
                    "copy-paste augmentation will be a no-op.", mode, self.copy_paste_prob
                )
            else:
                self._rare_class_patch_indices = self._index_rare_class_patches()
                if not self._rare_class_patch_indices:
                    logging.warning(
                        "GenDEBRIS(%s): no patches contain any of rare_classes=%s -- "
                        "copy-paste augmentation will be a no-op.", mode, self.rare_classes
                    )
                else:
                    logging.info(
                        "GenDEBRIS(%s): copy-paste augmentation enabled (prob=%.2f), "
                        "%d/%d patches available as donors for rare_classes=%s.",
                        mode, self.copy_paste_prob, len(self._rare_class_patch_indices),
                        len(self.masks), self.rare_classes
                    )

    def _index_rare_class_patches(self):
        """Indices of patches containing at least one rare class, for use as copy-paste donors."""
        return [i for i, m in enumerate(self.masks) if np.isin(m, self.rare_classes).any()]

    def _copy_paste_rare_classes(self, img, mask):
        """
        Paste rare-class pixels from a randomly chosen donor patch (one
        known to contain at least one rare class) onto (img, mask), at the
        same spatial positions. Image bands and mask label are copied
        together at each pasted pixel, so the two stay consistent -- this
        directly multiplies how often the model sees rare classes during
        training, on top of (not instead of) sample-level oversampling.

        Args:
            img (np.ndarray): (H, W, C) image, already NaN-imputed.
            mask (np.ndarray): (H, W) integer class labels.

        Returns:
            (np.ndarray, np.ndarray): augmented (img, mask), same shapes.
        """
        donor_idx = random.choice(self._rare_class_patch_indices)
        donor_img = np.moveaxis(self.images[donor_idx], 0, -1).astype(np.float32)
        donor_mask = self.masks[donor_idx]

        # Impute the donor's NaNs the same way __getitem__ does for the main image.
        donor_nan_mask = np.isnan(donor_img)
        donor_img[donor_nan_mask] = self.impute_nan[donor_nan_mask]

        paste_mask = np.isin(donor_mask, self.rare_classes)
        if not paste_mask.any():
            return img, mask  # shouldn't happen given _rare_class_patch_indices, but stay safe

        img = img.copy()
        mask = mask.copy()
        img[paste_mask] = donor_img[paste_mask]
        mask[paste_mask] = donor_mask[paste_mask]
        return img, mask

    def _spectral_jitter(self, img):
        """
        Multiply each band of a (C, H, W) tensor by an independent random
        factor in [1 - spectral_jitter_strength, 1 + spectral_jitter_strength],
        simulating realistic Sentinel-2 sensor/atmospheric variation. Cheap
        regularisation against overfitting exact band statistics on a
        694-patch training set.
        """
        num_bands = img.shape[0]
        factors = 1.0 + (torch.rand(num_bands, 1, 1) * 2 - 1) * self.spectral_jitter_strength
        return img * factors

    def __len__(self):
        return len(self.masks)

    def get_names(self):
        """Return the ROI identifiers."""
        return self.rois

    def compute_sample_weights(self, rare_classes, boost=5.0):
        """
        Compute a per-patch sampling weight, for use with
        torch.utils.data.WeightedRandomSampler, so that patches containing
        rare classes get drawn more often during training than patches
        that don't -- on top of (not instead of) any loss-level class
        weighting.

        Args:
            rare_classes (list[int]): 0-indexed class IDs (matching the
                values found in self.masks, i.e. in the same aggregated
                label space __getitem__ returns) considered rare/
                underperforming and worth oversampling.
            boost (float): extra weight added per rare class present in a
                patch. A patch with none of the rare classes present gets
                the base weight of 1.0; a patch with one rare class
                present gets 1.0 + boost; a patch with two gets
                1.0 + 2*boost, and so on.

        Returns:
            list[float]: one weight per patch, in the same order as
                self.masks / self.images (i.e. dataset index order).
        """
        weights = []
        for mask in self.masks:
            n_rare_present = sum(1 for c in rare_classes if np.any(mask == c))
            weights.append(1.0 + boost * n_rare_present)
        return weights

    def __getitem__(self, index):
        img = self.images[index].copy()
        mask = self.masks[index].copy()

        # Convert from (C, H, W) to (H, W, C) for transformations
        img = np.moveaxis(img, 0, -1).astype(np.float32)

        # Impute NaN values with band means
        nan_mask = np.isnan(img)
        img[nan_mask] = self.impute_nan[nan_mask]

        # ---- Copy-paste rare-class augmentation ----
        # Runs before the geometric transform below, so the pasted region
        # also gets rotated/flipped consistently along with the rest of
        # the patch.
        if self._rare_class_patch_indices and random.random() < self.copy_paste_prob:
            img, mask = self._copy_paste_rare_classes(img, mask)

        if self.transform is not None:
            # Concatenate mask as extra channel to apply same transform
            mask = mask[..., np.newaxis]
            stack = np.concatenate([img, mask], axis=-1).astype(np.float32)
            stack = self.transform(stack)
            # Separate image and mask
            img = stack[:-1, :, :]
            # Round to avoid interpolation artifacts in mask
            mask = stack[-1, :, :].round().long()
        else:
            # Convert to tensor if no transform
            img = torch.from_numpy(np.moveaxis(img, -1, 0))  # back to (C, H, W)
            mask = torch.from_numpy(mask)

        # ---- Spectral jitter ----
        # Image channels only -- applied after the mask has already been
        # split back out, so the (integer) class labels are never touched.
        if self.spectral_jitter_prob > 0 and random.random() < self.spectral_jitter_prob:
            img = self._spectral_jitter(img)

        if self.standardization is not None:
            img = self.standardization(img)

        return img, mask


class RandomRotationTransform:
    """Apply a random rotation from a list of angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


def gen_weights(class_distribution, c=1.02):
    """
    Compute class weights for cross-entropy loss.

    Args:
        class_distribution (torch.Tensor): Frequency of each class.
        c (float): Smoothing parameter.

    Returns:
        torch.Tensor: Class weights.
    """
    return 1.0 / torch.log(c + class_distribution)