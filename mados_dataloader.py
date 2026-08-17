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

VERIFIED (cross-checked against 2+ independent published sources, OR
directly confirmed against a real MADOS download):
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
  - Split file naming: CONFIRMED against a real MADOS download --
    <mados_path>/splits/{train,val,test}_X.txt, reusing MARIDA's exact
    naming convention (not train.txt/val.txt/test.txt as an earlier
    version of this file assumed).
  - The official scene-level train/val/test split also exists in
    MADOS's separately-distributed dataset.h5 (per-pixel spectral
    signature table, keys '/Train'/'/Validation'/'/Test', with a
    'Scene' column giving 174 total scenes: 96 train / 36 val / 42
    test) -- consistent with, and a useful cross-check against, the
    splits/*_X.txt files.

NOT VERIFIED -- please confirm on your actual download before trusting
results, same discipline as the confidence-raster naming issue earlier:
  - The exact file naming for MADOS patches under patches/ once
    downloaded and "stacked" (their own README describes a required
    `utils/stack_patches.py` step to combine raw per-band rasters into
    a single multiband GeoTIFF per patch -- this module assumes that
    step has already been run and produces `<name>.tif` / `<name>_cl.tif`
    pairs, matching the confirmed splits/*_X.txt naming pattern, but
    the patches/ folder's exact file naming itself has not yet been
    independently confirmed the way splits/ has).
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

import logging
import os
import random

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
    'Marine Debris', 'Dense Sargassum', 'Sparse Floating Algae', 'Natural Organic Material',
    'Ships', 'Oil Spills', 'Marine Water', 'Sediment-Laden Water',
    'Foam', 'Turbid Water', 'Shallow Water', 'Waves',
    'Oil Platforms', 'Jellyfish Aggregations', 'Sea Snot',
]
# CORRECTED against a direct screenshot of the actual published Table 5
# header row (Kikaki et al., 2024) -- the previous order here had "Oil
# Spills" in position 2 instead of position 6, shifting Dense Sargassum/
# Sparse Floating Algae/Natural Organic Material/Ships each one position
# earlier than they should be. This was a real, serious bug: it caused
# real Dense Sargassum pixels (raw mask value 2) to be silently dropped
# (mapped to the old, wrong "Oil Spills -> ignored" rule), and caused
# Sparse Floating Algae/Natural Organic Material/Ships pixels to be
# mistrained under the WRONG MARIDA class labels one position over, with
# real Oil Spill pixels (raw value 6) incorrectly taught to the model as
# "Ship". Any MADOS-trained checkpoint produced before this fix should be
# considered invalid for classes 2-6 and retrained.
# Crosswalk: MADOS class index (0-indexed into MADOS_LABELS) -> MARIDA
# class index (0-indexed into MARIDA_LABELS), or None if MADOS has no
# equivalent MARIDA class (these pixels become ignore_index=-1, i.e.
# excluded from training/evaluation entirely rather than guessed-at).
# Order CONFIRMED against a direct screenshot of the actual published
# Table 5 header row (Kikaki et al., 2024).
#
#   MADOS class (order)      -> MARIDA class            -> reasoning
#   1. Marine Debris          -> Marine Debris               direct match
#   2. Dense Sargassum         -> Dense Sargassum             direct match
#   3. Sparse Floating Algae   -> Sparse Sargassum            same concept, renamed
#   4. Natural Organic Material -> Natural Organic Material   direct match
#   5. Ships                    -> Ship                        direct match
#   6. Oil Spills                -> None (ignored)              no MARIDA equivalent
#   7. Marine Water              -> Marine Water                direct match
#   8. Sediment-Laden Water      -> Sediment-Laden Water        direct match
#   9. Foam                       -> Foam                        direct match
#   10. Turbid Water              -> Turbid Water                direct match
#   11. Shallow Water             -> Shallow Water               direct match
#   12. Waves                     -> Marine Water                matches MARIDA's own
#                                                                agg_to_water=True logic
#   13. Oil Platforms              -> None (ignored)              no MARIDA equivalent
#   14. Jellyfish Aggregations     -> None (ignored)              no MARIDA equivalent
#   15. Sea Snot                   -> None (ignored)              no MARIDA equivalent
MADOS_TO_MARIDA = {
    0: 0,    # Marine Debris -> Marine Debris
    1: 1,    # Dense Sargassum -> Dense Sargassum
    2: 2,    # Sparse Floating Algae -> Sparse Sargassum
    3: 3,    # Natural Organic Material -> Natural Organic Material
    4: 4,    # Ships -> Ship
    5: None,  # Oil Spills -> ignored (no MARIDA equivalent)
    6: 6,    # Marine Water -> Marine Water
    7: 7,    # Sediment-Laden Water -> Sediment-Laden Water
    8: 8,    # Foam -> Foam
    9: 9,    # Turbid Water -> Turbid Water
    10: 10,  # Shallow Water -> Shallow Water
    11: 6,   # Waves -> Marine Water (matches MARIDA's own agg_to_water logic)
    12: None,  # Oil Platforms -> ignored
    13: None,  # Jellyfish Aggregations -> ignored
    14: None,  # Sea Snot -> ignored
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
        rare_classes (list[int], optional): MARIDA-space class indices
            (post-crosswalk, e.g. 2 = Sparse Sargassum) to treat as
            copy-paste donors. Matches GenDEBRIS's --rare_classes
            convention so the two pipelines stay consistent.
        copy_paste_prob (float): probability of applying copy-paste to
            a given training sample, matching MADOS's own VSCP (Very
            Simple Copy-Paste) augmentation, confirmed to have been used
            by MariNeXt's authors (Kikaki et al., 2024) to help with
            exactly the same kind of severe rare-class imbalance you'd
            see with Sparse Sargassum on this dataset. Should only be
            set > 0 for the TRAIN split -- val/test must stay
            deterministic.
    """

    def __init__(self, mados_path, split='train', transform=None,
                 use_spectral_indices=False, use_texture_features=False,
                 standardization=None, splits_path=None,
                 rare_classes=None, copy_paste_prob=0.0):
        from osgeo import gdal

        self._gdal = gdal
        self.mados_path = mados_path
        self.transform = transform
        self.use_spectral_indices = use_spectral_indices
        self.use_texture_features = use_texture_features
        self.standardization = standardization
        self.n_raw_bands = len(BANDS_MEAN)

        # CONFIRMED against real data: after running MADOS's own
        # utils/stack_patches.py, the stacked (usable) image data lands in
        # a SIBLING folder named '<input>_nearest' (e.g. MADOS ->
        # MADOS_nearest), NOT inside the original MADOS/ folder in place.
        # The splits/ folder, however, is only ever created inside the
        # ORIGINAL (unstacked) MADOS/ folder. So --mados_path should point
        # at the STACKED data (e.g. .../MADOS_nearest) for image loading,
        # while splits_path defaults to the same directory but can be
        # overridden to point at the original MADOS/splits/ if the two
        # live in different places, which is the common case.
        splits_dir = splits_path if splits_path is not None else os.path.join(mados_path, 'splits')
        split_file = os.path.join(splits_dir, f'{split}_X.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Could not find {split_file}. This module assumes a MADOS split file layout "
                f"similar to MARIDA's -- check your actual MADOS download structure and adjust "
                f"this path (and the patch-file naming in __getitem__) to match."
            )
        with open(split_file) as f:
            self.rois = [line.strip() for line in f if line.strip()]

        self.rare_classes = list(rare_classes) if rare_classes else []
        self.copy_paste_prob = copy_paste_prob
        self._rare_class_roi_indices = []
        if self.copy_paste_prob > 0:
            if not self.rare_classes:
                logging.warning(
                    "MADOSDataset(%s): copy_paste_prob=%.2f but rare_classes is empty -- "
                    "copy-paste augmentation will be a no-op.", split, self.copy_paste_prob
                )
            else:
                self._rare_class_roi_indices = self._index_rare_class_rois()
                if not self._rare_class_roi_indices:
                    logging.warning(
                        "MADOSDataset(%s): no ROIs contain any of rare_classes=%s (in MARIDA "
                        "label space) -- copy-paste augmentation will be a no-op.",
                        split, self.rare_classes
                    )
                else:
                    logging.info(
                        "MADOSDataset(%s): copy-paste (VSCP-style) augmentation enabled "
                        "(prob=%.2f), %d/%d ROIs available as donors for rare_classes=%s.",
                        split, self.copy_paste_prob, len(self._rare_class_roi_indices),
                        len(self.rois), self.rare_classes
                    )

    def __len__(self):
        return len(self.rois)

    def _roi_to_paths(self, roi):
        """Shared helper: ROI name -> (img_path, mask_path), matching __getitem__'s logic."""
        scene_id, crop_id = roi.rsplit('_', 1)
        img_path = os.path.join(self.mados_path, scene_id, f'{scene_id}_L2R_rhorc_{crop_id}.tif')
        mask_path = os.path.join(self.mados_path, scene_id, f'{scene_id}_L2R_cl_{crop_id}.tif')
        return img_path, mask_path

    def _index_rare_class_rois(self):
        """
        One-time scan (at construction) over every ROI's mask file to find
        which ones contain at least one of self.rare_classes, AFTER
        remapping to MARIDA's label space -- so --rare_classes uses the
        same class-index convention as the rest of the pipeline (e.g. 2 =
        Sparse Sargassum), not MADOS's raw 15-class codes. Unlike
        GenDEBRIS, MADOSDataset doesn't preload every mask into memory, so
        this reads each mask file once, here, rather than reusing an
        already-loaded array.
        """
        donor_indices = []
        for i, roi in enumerate(self.rois):
            _, mask_path = self._roi_to_paths(roi)
            ds = self._gdal.Open(mask_path)
            if ds is None:
                continue
            raw_mask = ds.ReadAsArray().astype(np.int64)
            ds = None
            remapped = remap_mados_mask(raw_mask)
            if np.isin(remapped, self.rare_classes).any():
                donor_indices.append(i)
        return donor_indices

    def _copy_paste_rare_classes(self, img, mask):
        """
        Paste rare-class pixels from a randomly chosen donor ROI (one
        known to contain at least one rare class, see
        _index_rare_class_rois) onto (img, mask), at the same spatial
        positions -- same VSCP-style idea as GenDEBRIS's copy-paste for
        MARIDA. The donor is loaded fresh here (lazily), matching
        MADOSDataset's overall lazy-loading design.

        Args:
            img (np.ndarray): (H, W, C) image, already NaN-imputed and
                resized to 256x256, with any extra channels (spectral
                indices/texture) already appended if enabled.
            mask (np.ndarray): (H, W) MARIDA-space integer class labels
                (already remapped + resized), -1 = ignore.

        Returns:
            (np.ndarray, np.ndarray): augmented (img, mask), same shapes.
        """
        donor_idx = random.choice(self._rare_class_roi_indices)
        donor_roi = self.rois[donor_idx]
        donor_img_path, donor_mask_path = self._roi_to_paths(donor_roi)

        ds = self._gdal.Open(donor_img_path)
        donor_img = ds.ReadAsArray().astype(np.float32)  # (C, H, W)
        ds = None
        donor_img = np.moveaxis(donor_img, 0, -1)  # (H, W, C)

        ds = self._gdal.Open(donor_mask_path)
        donor_mask_raw = ds.ReadAsArray().astype(np.int64)
        ds = None
        donor_mask = remap_mados_mask(donor_mask_raw)

        donor_img, donor_mask = resize_to_256(donor_img, donor_mask)

        donor_nan_mask = np.isnan(donor_img)
        band_means = np.tile(BANDS_MEAN, (donor_img.shape[0], donor_img.shape[1], 1))
        donor_img[donor_nan_mask] = band_means[donor_nan_mask]

        if self.use_spectral_indices or self.use_texture_features:
            parts = []
            if self.use_spectral_indices:
                parts.append(compute_spectral_indices(donor_img))
            if self.use_texture_features:
                parts.append(compute_texture_features(donor_img))
            donor_img = np.concatenate([donor_img] + parts, axis=-1)

        paste_mask = np.isin(donor_mask, self.rare_classes)
        if not paste_mask.any():
            return img, mask  # shouldn't happen given _rare_class_roi_indices, but stay safe

        img = img.copy()
        mask = mask.copy()
        img[paste_mask] = donor_img[paste_mask]
        mask[paste_mask] = donor_mask[paste_mask]
        return img, mask

    def __getitem__(self, index):
        roi = self.rois[index]
        img_path, mask_path = self._roi_to_paths(roi)

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

        # ---- Copy-paste (VSCP-style) rare-class augmentation ----
        # Runs before the geometric transform below, so the pasted region
        # also gets rotated/flipped consistently along with the rest of
        # the patch -- identical ordering to GenDEBRIS's MARIDA pipeline.
        if self._rare_class_roi_indices and random.random() < self.copy_paste_prob:
            img, mask = self._copy_paste_rare_classes(img, mask)

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