# -*- coding: utf-8 -*-
"""
mados_dataloader.py

Shared MADOS dataset logic, used by BOTH evaluate_on_mados.py (evaluate
an existing MARIDA checkpoint on MADOS) and train_on_mados.py (train a
new model from scratch on MADOS only). Keeping this in one place means
the class crosswalk, resize logic, and channel handling can never drift
between the two scripts.

Everything here reports in MARIDA's 11-class label space (see
MARIDA_LABELS / MADOS_TO_MARIDA below) so a MADOS-trained model can be
evaluated with the exact same metrics/report code as a MARIDA-trained
one.

======================================================================
VERIFIED vs. UNVERIFIED -- read before trusting any output
======================================================================

VERIFIED (cross-checked against 2+ independent published sources):
  - MADOS's 15-class list and their 1-indexed order (see MADOS_LABELS
    below) -- confirmed by a Frontiers paper doing MARIDA+MADOS merging
    (which explicitly lists all 15 in order) AND independently by a
    second paper that references "Class 1" = Marine Debris and
    "Class 9" = Foam, matching this exact ordering.
  - MADOS uses the SAME 11 Sentinel-2 bands as MARIDA (B1-B8A, B11, B12,
    excluding Vapour/B9 and Cirrus/B10) -- confirmed via the PANGAEA
    foundation-model benchmark paper. No band-reordering needed.
  - Patch size: MADOS patches are 240x240 (not MARIDA's 256x256) --
    confirmed via PANGAEA. This module center-crops/pads to 256x256 to
    match your model's expected input; adjust MADOS_PATCH_SIZE below if
    this is wrong for your actual download.

NOT VERIFIED -- please confirm on your actual download before trusting
results, same discipline as the confidence-raster naming issue earlier:
  - The exact file/folder naming convention for MADOS patches once
    downloaded and "stacked" (their own README describes a required
    `utils/stack_patches.py` step to combine raw per-band rasters into
    a single multiband GeoTIFF per patch -- this module assumes that
    step has already been run and produces `<name>.tif` / `<name>_cl.tif`
    pairs in the same style as MARIDA's dataloader, but this is an
    ASSUMPTION based on the MARIDA-derived codebase pattern, not a
    confirmed file listing).
  - Whether MADOS's raster mask values are 1-indexed the same way
    MARIDA's are.
  - Band statistics (BANDS_MEAN/BANDS_STD): reused directly from
    MARIDA's dataloader.py, assuming MADOS's Sentinel-2 reflectance
    values are on the same processing/scale as MARIDA's (both are
    described as multispectral S2 data with similar bands, but this has
    NOT been independently confirmed by inspecting real MADOS pixel
    statistics).

IMPORTANT -- MADOS has NO "Clouds" class:
MARIDA's class 5 (Clouds) does not appear anywhere in MADOS's 15-class
list. A model trained ONLY on MADOS will NEVER see a single Clouds
example during training -- this is a structural limitation of
MADOS-only training, not a bug. Both evaluate_on_mados.py and
train_on_mados.py exclude Clouds from macro-average computations
accordingly (checkpoint selection, reported metrics), rather than
silently scoring it as 0% or crashing.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader import (
    BANDS_MEAN, BANDS_STD,
    compute_spectral_indices, compute_texture_features,
)

MADOS_PATCH_SIZE = 240  # PANGAEA-confirmed; change if your download differs

# MARIDA's 11 output classes, in the exact order your model was trained
# to predict (matches ALL_LABELS[:11] / agg_to_water=True in dataloader.py).
MARIDA_LABELS = [
    'Marine Debris', 'Dense Sargassum', 'Sparse Sargassum',
    'Natural Organic Material', 'Ship', 'Clouds', 'Marine Water',
    'Sediment-Laden Water', 'Foam', 'Turbid Water', 'Shallow Water',
]

# MADOS's 15 classes, 1-indexed as they appear in the raw mask rasters
# (index 0 in this list = mask value 1, etc.) -- see VERIFIED note above.
MADOS_LABELS = [
    'Marine Debris', 'Oil Spills', 'Dense Sargassum', 'Sparse Floating Algae',
    'Natural Organic Material', 'Ships', 'Marine Water', 'Sediment-Laden Water',
    'Foam', 'Turbid Water', 'Shallow Water', 'Waves and Wakes',
    'Oil Platforms', 'Jellyfish Aggregations', 'Sea Snot',
]

# Crosswalk: MADOS class index (0-indexed into MADOS_LABELS) -> MARIDA
# class index (0-indexed into MARIDA_LABELS), or None if MADOS has no
# equivalent MARIDA class (these pixels become ignore_index=-1, i.e.
# excluded from training/evaluation entirely rather than guessed-at).
#
#   MADOS class            -> MARIDA class            -> reasoning
#   Marine Debris          -> Marine Debris               direct match
#   Oil Spills              -> None (ignored)              no MARIDA equivalent
#   Dense Sargassum          -> Dense Sargassum             direct match
#   Sparse Floating Algae    -> Sparse Sargassum            same concept, renamed
#   Natural Organic Material -> Natural Organic Material    direct match
#   Ships                     -> Ship                        direct match
#   Marine Water             -> Marine Water                direct match
#   Sediment-Laden Water     -> Sediment-Laden Water        direct match
#   Foam                       -> Foam                        direct match
#   Turbid Water              -> Turbid Water                direct match
#   Shallow Water             -> Shallow Water               direct match
#   Waves and Wakes           -> Marine Water                matches MARIDA's own
#                                                            agg_to_water=True logic
#   Oil Platforms              -> None (ignored)              no MARIDA equivalent
#   Jellyfish Aggregations     -> None (ignored)              no MARIDA equivalent
#   Sea Snot                   -> None (ignored)              no MARIDA equivalent
MADOS_TO_MARIDA = {
    0: 0, 1: None, 2: 1, 3: 2, 4: 3, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 6, 12: None, 13: None, 14: None,
}

# MARIDA classes that MADOS cannot test/train at all (no equivalent in
# its label set) -- excluded from macro averages computed on MADOS data.
MARIDA_CLASSES_NOT_IN_MADOS = ['Clouds']


def remap_mados_mask(mados_mask):
    """
    Convert a raw MADOS mask (1-indexed class codes) into MARIDA's
    0-indexed 11-class label space, using MADOS_TO_MARIDA. Unmapped
    classes and anything outside the valid MADOS range become -1
    (ignore_index), matching how MARIDA's own masks already encode
    "don't train/evaluate on this pixel".
    """
    remapped = np.full_like(mados_mask, -1)
    for mados_idx, marida_idx in MADOS_TO_MARIDA.items():
        if marida_idx is not None:
            remapped[mados_mask == (mados_idx + 1)] = marida_idx  # +1: MADOS masks are 1-indexed
    return remapped


def resize_to_256(img, mask, target=256):
    """Center pad-then-crop an (H, W, C) image and (H, W) mask to target x target.
    Padding uses reflect for the image and -1 (ignore_index) for the mask,
    so padded regions are never trained/evaluated on."""
    h, w = mask.shape
    if h == target and w == target:
        return img, mask
    pad_h, pad_w = max(0, target - h), max(0, target - w)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=-1)
    h2, w2 = mask.shape[:2]
    if h2 > target or w2 > target:
        top = (h2 - target) // 2
        left = (w2 - target) // 2
        img = img[top:top + target, left:left + target]
        mask = mask[top:top + target, left:left + target]
    return img, mask


class MADOSDataset(Dataset):
    """
    MADOS loader: reads already-stacked <name>.tif / <name>_cl.tif pairs
    (see the NOT VERIFIED note at the top of this file), remaps labels
    to MARIDA's 11-class space via remap_mados_mask, optionally computes
    the same spectral-index/texture extra channels used elsewhere in
    this project, and center-crops/pads to 256x256.

    Args:
        mados_path (str): root MADOS directory (contains patches/, splits/).
        split (str): 'train', 'val', or 'test' -- reads
            <mados_path>/splits/<split>.txt.
        transform: a torchvision transforms.Compose, applied to the
            (H, W, C+1) image+mask stack -- IDENTICAL pattern to
            GenDEBRIS in dataloader.py. Pass
            transforms.Compose([transforms.ToTensor(),
            RandomRotationTransform([-90,0,90,180]),
            transforms.RandomHorizontalFlip()]) for training (matching
            train_swin_unetv2.py's train-time augmentation exactly), or
            transforms.Compose([transforms.ToTensor()]) for
            deterministic val/test. Required (not optional), since
            ToTensor is what performs the numpy->tensor + HWC->CHW
            conversion.
    """

    def __init__(self, mados_path, split='train', transform=None,
                 use_spectral_indices=False, use_texture_features=False,
                 standardization=None):
        from osgeo import gdal

        self._gdal = gdal
        self.mados_path = mados_path
        self.transform = transform
        self.use_spectral_indices = use_spectral_indices
        self.use_texture_features = use_texture_features
        self.standardization = standardization
        self.n_raw_bands = len(BANDS_MEAN)

        split_file = os.path.join(mados_path, 'splits', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Could not find {split_file}. This module assumes a MADOS split file layout "
                f"similar to MARIDA's -- check your actual MADOS download structure and adjust "
                f"this path (and the patch-file naming in __getitem__) to match."
            )
        with open(split_file) as f:
            self.rois = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.rois)

    def __getitem__(self, index):
        roi = self.rois[index]
        img_path = os.path.join(self.mados_path, 'patches', f'{roi}.tif')
        mask_path = os.path.join(self.mados_path, 'patches', f'{roi}_cl.tif')

        ds = self._gdal.Open(img_path)
        img = ds.ReadAsArray().astype(np.float32)  # (C, H, W)
        ds = None

        ds = self._gdal.Open(mask_path)
        mask = ds.ReadAsArray().astype(np.int64)  # (H, W), MADOS's raw 1-indexed codes
        ds = None

        mask = remap_mados_mask(mask)

        img = np.moveaxis(img, 0, -1)  # (H, W, C)
        img, mask = resize_to_256(img, mask)

        nan_mask = np.isnan(img)
        band_means = np.tile(BANDS_MEAN, (img.shape[0], img.shape[1], 1))
        img[nan_mask] = band_means[nan_mask]

        if self.use_spectral_indices or self.use_texture_features:
            parts = []
            if self.use_spectral_indices:
                parts.append(compute_spectral_indices(img))
            if self.use_texture_features:
                parts.append(compute_texture_features(img))
            img = np.concatenate([img] + parts, axis=-1)

        if self.transform is not None:
            # Concatenate mask as an extra channel, exactly like
            # GenDEBRIS, so the transform's ToTensor + rotation + flip
            # apply identically to image and mask together, keeping
            # them spatially aligned.
            mask_ch = mask[..., np.newaxis]
            stack = np.concatenate([img, mask_ch], axis=-1).astype(np.float32)
            stack = self.transform(stack)
            img_t = stack[:-1, :, :]
            mask = stack[-1, :, :].round().long()
        else:
            img_t = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
            mask = torch.from_numpy(mask)

        if self.use_spectral_indices or self.use_texture_features:
            raw = img_t[:self.n_raw_bands]
            extra = img_t[self.n_raw_bands:]
            if self.standardization is not None:
                raw = self.standardization(raw)
            img_t = torch.cat([raw, extra], dim=0)
        elif self.standardization is not None:
            img_t = self.standardization(img_t)

        return img_t, mask
