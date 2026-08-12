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
from scipy import ndimage
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

# Band order for the 11 raw Sentinel-2 bands used throughout this file and
# by GenDEBRIS (matches MARIDA's ACOLITE-processed bands, Vapour/B9 and
# Cirrus/B10 excluded): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
# Index into the last axis of a (H, W, 11) or first axis of a (11, H, W)
# raw-band array/tensor:
_B1, _B2, _B3, _B4, _B5, _B6, _B7, _B8, _B8A, _B11, _B12 = range(11)

# Approximate Sentinel-2 central wavelengths (nm) for the bands used below.
_WL = {'B4': 665.0, 'B6': 740.0, 'B8': 842.0, 'B11': 1610.0}

_SPECTRAL_INDEX_EPS = 1e-6
SPECTRAL_INDEX_NAMES = ['NDVI', 'NDWI', 'NDMI', 'BSI', 'FAI', 'FDI']


def compute_spectral_indices(img):
    """
    Compute 6 spectral indices from an (H, W, 11) raw-reflectance array (or
    (11, H, W); auto-detected), in the same band order as BANDS_MEAN/STD.
    These are the same class of hand-engineered features (spectral indices
    + implicitly, texture via CON in the original MARIDA study) that gave
    MARIDA's Random Forest baseline its edge over from-scratch deep models
    trained on raw bands alone -- stacking them as extra input channels
    lets the network use the same information directly instead of having
    to re-derive it from ~700 training patches.

    Indices (all bounded, roughly in [-1, 1] except FAI/FDI which are
    small band-difference values, typically within [-0.1, 0.1]):
        NDVI = (B8 - B4) / (B8 + B4)                          [vegetation]
        NDWI = (B3 - B8) / (B3 + B8)          [McFeeters 1996; water]
        NDMI = (B8 - B11) / (B8 + B11)                        [moisture]
        BSI  = ((B11+B4) - (B8+B2)) / ((B11+B4) + (B8+B2))    [bare soil]
        FAI  = B8 - [B4 + (B11-B4) * (842-665)/(1610-665)]
                                          [Hu 2009; floating algae]
        FDI  = B8 - [B6 + (B11-B6) * (842-740)/(1610-740) * 10]
                                          [Biermann et al. 2020; floating debris]

    Args:
        img (np.ndarray): (H, W, 11) or (11, H, W) raw-band array.

    Returns:
        np.ndarray: indices in the same layout as the input
        ((H, W, 6) or (6, H, W)), order matching SPECTRAL_INDEX_NAMES.
    """
    channels_last = img.shape[-1] == 11
    if not channels_last:
        img = np.moveaxis(img, 0, -1)  # -> (H, W, 11)

    eps = _SPECTRAL_INDEX_EPS
    b2, b3, b4 = img[..., _B2], img[..., _B3], img[..., _B4]
    b6, b8, b11 = img[..., _B6], img[..., _B8], img[..., _B11]

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    ndmi = (b8 - b11) / (b8 + b11 + eps)
    bsi = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)

    fai_baseline = b4 + (b11 - b4) * (_WL['B8'] - _WL['B4']) / (_WL['B11'] - _WL['B4'])
    fai = b8 - fai_baseline

    fdi_baseline = b6 + (b11 - b6) * (_WL['B8'] - _WL['B6']) / (_WL['B11'] - _WL['B6']) * 10.0
    fdi = b8 - fdi_baseline

    indices = np.stack([ndvi, ndwi, ndmi, bsi, fai, fdi], axis=-1).astype(np.float32)

    if not channels_last:
        indices = np.moveaxis(indices, -1, 0)  # -> (6, H, W)
    return indices


TEXTURE_FEATURE_NAMES = ['local_std', 'gradient_magnitude']
_TEXTURE_WINDOW = 13  # matches the 13x13 GLCM window used in the MARIDA paper


def compute_texture_features(img, window=_TEXTURE_WINDOW):
    """
    Compute 2 fast texture features from an (H, W, 11) or (11, H, W) raw
    band array: local standard deviation and local gradient magnitude,
    over a `window`x`window` neighbourhood (13x13, matching the GLCM
    window MARIDA used) on a luminance composite of the visible bands.

    This is a *proxy* for the GLCM texture features (Contrast,
    Dissimilarity, Homogeneity, Energy, Correlation, ASM) used in the
    original MARIDA Random Forest -- not a re-implementation. True GLCM
    requires quantizing the image and counting pixel co-occurrence pairs
    within a sliding window, which is too slow to run on-the-fly once per
    sample per epoch. Local std and gradient magnitude are cheap
    (vectorised, O(HW)) approximations that capture the same underlying
    idea -- "how rough/high-contrast is the neighbourhood around this
    pixel" -- which MARIDA's own feature-importance analysis (Fig. 6 of
    Kikaki et al., 2022) found to be the single most informative feature
    (CON) for their winning RF variant.

    Args:
        img (np.ndarray): (H, W, 11) or (11, H, W) raw-band array.
        window (int): side length of the local window (odd number).

    Returns:
        np.ndarray: texture features in the same layout as the input
        ((H, W, 2) or (2, H, W)), order matching TEXTURE_FEATURE_NAMES.
    """
    channels_last = img.shape[-1] == 11
    if not channels_last:
        img = np.moveaxis(img, 0, -1)  # -> (H, W, 11)

    # Luminance composite from the visible bands (B4=Red, B3=Green, B2=Blue),
    # matching MARIDA's "Rayleigh corrected RGB composites converted to
    # grayscale" description for their GLCM computation.
    gray = (0.299 * img[..., _B4] + 0.587 * img[..., _B3] + 0.114 * img[..., _B2]).astype(np.float32)

    # Local standard deviation via the identity Var(X) = E[X^2] - E[X]^2,
    # computed with fast windowed means (uniform_filter is a separable
    # box-filter convolution, O(HW) regardless of window size).
    local_mean = ndimage.uniform_filter(gray, size=window, mode='reflect')
    local_sq_mean = ndimage.uniform_filter(gray * gray, size=window, mode='reflect')
    local_var = np.clip(local_sq_mean - local_mean ** 2, 0.0, None)  # guard tiny negative FP error
    local_std = np.sqrt(local_var).astype(np.float32)

    # Sobel gradient magnitude -- a standard, cheap edge/contrast measure.
    gx = ndimage.sobel(gray, axis=1, mode='reflect')
    gy = ndimage.sobel(gray, axis=0, mode='reflect')
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    features = np.stack([local_std, gradient_magnitude], axis=-1)

    if not channels_last:
        features = np.moveaxis(features, -1, 0)  # -> (2, H, W)
    return features


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
        use_spectral_indices (bool): If True, append 6 spectral indices
            (NDVI, NDWI, NDMI, BSI, FAI, FDI -- see compute_spectral_indices)
            as extra input channels after the 11 raw bands, giving the
            model direct access to the same class of hand-engineered
            features that gave MARIDA's own Random Forest baseline an
            edge over from-scratch deep models trained on raw bands
            alone. Adds 6 to self.num_channels. Must be set consistently
            across train/val/test splits used together.
        use_texture_features (bool): If True, append 2 fast texture
            features (local standard deviation + gradient magnitude over
            a 13x13 window -- see compute_texture_features) as extra
            input channels. A cheap proxy for the GLCM texture features
            (Contrast, Energy, etc.) that MARIDA's own feature-importance
            analysis found to be individually more informative than any
            single spectral index for their winning RF variant. Adds 2
            to self.num_channels. Must be set consistently across
            train/val/test splits used together.
    """

    def __init__(self, mode='train', transform=None, standardization=None,
                 path=DATASET_PATH, agg_to_water=True, rare_classes=None,
                 copy_paste_prob=0.0, spectral_jitter_prob=0.0,
                 spectral_jitter_strength=0.05, use_spectral_indices=False,
                 use_texture_features=False,
                 splits_dir='splits'):
        super().__init__()

        # splits_dir: 'splits' (default) uses MARIDA's official ~50/24/26
        # split. Pass e.g. 'splits_80_10_10' (see create_custom_split.py)
        # to use a custom ratio instead -- NOTE this breaks direct
        # comparability with MARIDA's own reported numbers, which use
        # their official split.
        split_file = os.path.join(path, splits_dir, f'{mode}_X.txt')
        self.rois = np.genfromtxt(split_file, dtype='str')

        self.images = []       # list of image arrays (C, H, W)
        self.masks = []        # list of mask arrays (H, W)

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

        self.use_spectral_indices = use_spectral_indices
        self.use_texture_features = use_texture_features
        self.n_raw_bands = len(BANDS_MEAN)  # 11
        n_extra = (len(SPECTRAL_INDEX_NAMES) if use_spectral_indices else 0) \
            + (len(TEXTURE_FEATURE_NAMES) if use_texture_features else 0)
        self.num_channels = self.n_raw_bands + n_extra
        if use_spectral_indices or use_texture_features:
            extra_names = (SPECTRAL_INDEX_NAMES if use_spectral_indices else []) \
                + (TEXTURE_FEATURE_NAMES if use_texture_features else [])
            logging.info(
                "GenDEBRIS(%s): extra input channels enabled (%s), num_channels=%d (%d raw + %d extra).",
                mode, extra_names, self.num_channels, self.n_raw_bands, n_extra
            )

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

    def _compute_extra_channels(self, raw_img):
        """
        Build the extra-channel block (spectral indices and/or texture
        features, per whichever flags are enabled) for an (H, W, 11) raw,
        NaN-imputed band array. Returns an (H, W, 0-8) array (0 wide if
        neither is enabled) so callers can always do
        `np.concatenate([raw_img, extra], axis=-1)` uniformly.
        """
        parts = []
        if self.use_spectral_indices:
            parts.append(compute_spectral_indices(raw_img))
        if self.use_texture_features:
            parts.append(compute_texture_features(raw_img))
        if not parts:
            return np.zeros(raw_img.shape[:2] + (0,), dtype=np.float32)
        return np.concatenate(parts, axis=-1)

    def _copy_paste_rare_classes(self, img, mask):
        """
        Paste rare-class pixels from a randomly chosen donor patch (one
        known to contain at least one rare class) onto (img, mask), at the
        same spatial positions. Image bands and mask label are copied
        together at each pasted pixel, so the two stay consistent -- this
        directly multiplies how often the model sees rare classes during
        training, on top of (not instead of) sample-level oversampling.

        Args:
            img (np.ndarray): (H, W, C) image, already NaN-imputed. C is
                11 raw bands, plus any extra channels already appended by
                the caller (spectral indices and/or texture features) --
                the donor is built to match via _compute_extra_channels.
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

        if self.use_spectral_indices or self.use_texture_features:
            # Match the recipient's channel layout so the paste-mask
            # assignment below has matching channel counts.
            donor_extra = self._compute_extra_channels(donor_img)
            donor_img = np.concatenate([donor_img, donor_extra], axis=-1)

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

        # ---- Extra channels: spectral indices and/or texture features ----
        # Computed from raw reflectance (before copy-paste/transform) and
        # concatenated as extra channels, so they flow through copy-paste
        # and the geometric transform (rotation/flip) exactly like the raw
        # bands and stay spatially aligned. Split back out below, after the
        # transform, so standardization is only ever applied to the raw
        # bands it was fit for.
        if self.use_spectral_indices or self.use_texture_features:
            extra = self._compute_extra_channels(img)
            img = np.concatenate([img, extra], axis=-1)

        # ---- Copy-paste rare-class augmentation ----
        # Runs before the geometric transform below, so the pasted region
        # also gets rotated/flipped consistently along with the rest of
        # the patch.
        if self._rare_class_patch_indices and random.random() < self.copy_paste_prob:
            img, mask = self._copy_paste_rare_classes(img, mask)

        if self.transform is not None:
            # Concatenate mask as extra channel to apply same transform
            mask_ch = mask[..., np.newaxis]
            stack = np.concatenate([img, mask_ch], axis=-1).astype(np.float32)
            stack = self.transform(stack)
            # Separate image and mask
            img = stack[:-1, :, :]
            # Round to avoid interpolation artifacts in mask
            mask = stack[-1, :, :].round().long()
        else:
            # Convert to tensor if no transform
            img = torch.from_numpy(np.moveaxis(img, -1, 0))  # back to (C, H, W)
            mask = torch.from_numpy(mask)

        # ---- Split raw bands / spectral indices, jitter + standardize raw only ----
        # Spectral indices are already-bounded ratios (roughly [-1, 1], or
        # small band differences for FAI/FDI, or texture magnitudes) computed
        # from *raw* reflectance -- they must not be passed through
        # standardization fit on raw-band statistics, and jitter (a raw-
        # sensor-noise simulation) is only meaningful on the raw bands.
        if self.use_spectral_indices or self.use_texture_features:
            raw = img[:self.n_raw_bands]
            extra = img[self.n_raw_bands:]

            if self.spectral_jitter_prob > 0 and random.random() < self.spectral_jitter_prob:
                raw = self._spectral_jitter(raw)
            if self.standardization is not None:
                raw = self.standardization(raw)

            img = torch.cat([raw, extra], dim=0)
        else:
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