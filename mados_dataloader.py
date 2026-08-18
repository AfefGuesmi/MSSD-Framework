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


def remap_native_predictions_to_marida(pred_15class):
    """
    Given predictions made in MADOS's native 15-class space (0-indexed
    class IDs, as produced by a model trained with native_15_classes=True),
    remap them down to MARIDA's 11-class space for evaluation/reporting,
    using the exact same crosswalk as remap_mados_mask. Predictions of a
    class with no MARIDA equivalent (Oil Spills, Oil Platforms, Jellyfish,
    Sea Snot) become -1 (excluded from evaluation), matching how those
    classes are already excluded for a MARIDA-space-trained model.
    """
    return remap_mados_mask(pred_15class + 1)  # +1: remap_mados_mask expects 1-indexed raw codes


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
        use_glcm_texture (bool): load TRUE precomputed GLCM texture
            features (Contrast, Dissimilarity, Homogeneity, Energy,
            Correlation, ASM -- 6 channels) from
            <scene>_L2R_glcm_<crop>.tif, produced by running
            precompute_mados_glcm.py FIRST. This is mutually exclusive
            with use_texture_features (the fast live-computed std/
            gradient proxy) -- if both are True, GLCM takes priority
            and the fast proxy is skipped, since GLCM is strictly the
            more faithful (if slower to produce) feature set.
    """

    def __init__(self, mados_path, split='train', transform=None,
                 use_spectral_indices=False, use_texture_features=False,
                 standardization=None, splits_path=None, use_glcm_texture=False,
                 spectral_jitter_prob=0.0, spectral_jitter_strength=0.05,
                 native_15_classes=False):
        from osgeo import gdal

        self._gdal = gdal
        self.mados_path = mados_path
        self.transform = transform
        self.use_spectral_indices = use_spectral_indices
        self.use_glcm_texture = use_glcm_texture
        # GLCM (if enabled) replaces the fast proxy rather than stacking
        # both -- they're two versions of the same idea, not complementary.
        self.use_texture_features = use_texture_features and not use_glcm_texture
        self.spectral_jitter_prob = spectral_jitter_prob
        self.spectral_jitter_strength = spectral_jitter_strength
        self.standardization = standardization
        self.n_raw_bands = len(BANDS_MEAN)
        # native_15_classes: train on MADOS's own 15-class taxonomy
        # directly (Oil Spills, Oil Platforms, Jellyfish, Sea Snot no
        # longer thrown away as ignore_index), instead of remapping every
        # mask down to MARIDA's 11 classes. Gives the model real learning
        # signal from the ~4 extra classes' pixels too, which otherwise
        # contribute nothing to the loss at all. Evaluation-time code
        # (train_on_mados.py / evaluate_on_mados.py) still remaps
        # PREDICTIONS down to MARIDA's 11-class space for reporting, so
        # results stay comparable to non-native-15 runs.
        self.native_15_classes = native_15_classes

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

    def __len__(self):
        return len(self.rois)

    def _spectral_jitter(self, img):
        """
        Multiply each band of a (C, H, W) tensor by an independent random
        factor in [1 - spectral_jitter_strength, 1 + spectral_jitter_strength],
        simulating realistic Sentinel-2 sensor/atmospheric variation. Same
        logic as GenDEBRIS._spectral_jitter in dataloader.py (MARIDA).
        """
        num_bands = img.shape[0]
        factors = 1.0 + (torch.rand(num_bands, 1, 1) * 2 - 1) * self.spectral_jitter_strength
        return img * factors

    def compute_sample_weights(self, rare_classes, boost=5.0):
        """
        Compute a per-ROI sampling weight, for use with
        torch.utils.data.WeightedRandomSampler, so that patches containing
        rare classes get drawn more often during training than patches
        that don't -- on top of (not instead of) any loss-level class
        weighting from gen_weights(). This is a SAFER, more established
        alternative to copy-paste augmentation: it only changes how often
        a real, unmodified patch is drawn, never synthesizes or alters
        image content the way copy-paste does.

        Unlike GenDEBRIS (which has every mask preloaded in self.masks),
        MADOSDataset loads lazily, so this does a one-time scan over every
        ROI's mask file here, applying the same crosswalk remapping
        (remap_mados_mask) so rare_classes are specified in MARIDA's
        label space (e.g. 2 = Sparse Sargassum), matching the rest of
        this pipeline's convention.

        Args:
            rare_classes (list[int]): 0-indexed MARIDA-space class IDs
                considered rare/underperforming and worth oversampling.
            boost (float): extra weight added per rare class present in a
                patch. A patch with none of the rare classes present gets
                the base weight of 1.0; a patch with one rare class
                present gets 1.0 + boost; a patch with two gets
                1.0 + 2*boost, and so on.

        Returns:
            list[float]: one weight per ROI, in the same order as
                self.rois (i.e. dataset index order).
        """
        weights = []
        for roi in self.rois:
            scene_id, crop_id = roi.rsplit('_', 1)
            mask_path = os.path.join(self.mados_path, scene_id, f'{scene_id}_L2R_cl_{crop_id}.tif')
            ds = self._gdal.Open(mask_path)
            if ds is None:
                weights.append(1.0)  # can't read it; don't let it crash the whole scan
                continue
            raw_mask = ds.ReadAsArray().astype(np.int64)
            ds = None
            remapped = remap_mados_mask(raw_mask)
            n_rare_present = sum(1 for c in rare_classes if np.any(remapped == c))
            weights.append(1.0 + boost * n_rare_present)
        return weights

    def __getitem__(self, index):
        roi = self.rois[index]
        # CONFIRMED against real stacked data: ROI names are
        # 'Scene_<scene_id>_<crop_id>' (e.g. 'Scene_54_30'). The stacked
        # files live under Scene_<scene_id>/ with product-specific infixes:
        #   Scene_<scene_id>_L2R_rhorc_<crop_id>.tif  -- stacked multiband image
        #   Scene_<scene_id>_L2R_cl_<crop_id>.tif     -- class mask
        #   Scene_<scene_id>_L2R_glcm_<crop_id>.tif   -- precomputed GLCM (6-band),
        #                                                 only if precompute_mados_glcm.py has run
        # NOT a flat 'patches/<roi>.tif' layout like MARIDA's.
        scene_id, crop_id = roi.rsplit('_', 1)
        img_path = os.path.join(self.mados_path, scene_id, f'{scene_id}_L2R_rhorc_{crop_id}.tif')
        mask_path = os.path.join(self.mados_path, scene_id, f'{scene_id}_L2R_cl_{crop_id}.tif')

        ds = self._gdal.Open(img_path)
        img = ds.ReadAsArray().astype(np.float32)  # (C, H, W)
        ds = None

        ds = self._gdal.Open(mask_path)
        mask = ds.ReadAsArray().astype(np.int64)  # (H, W), MADOS's raw 1-indexed codes
        ds = None

        if self.native_15_classes:
            # Keep MADOS's own 0-indexed 15-class labels directly, no
            # crosswalk applied -- only shift 1-indexed raw codes to
            # 0-indexed (1..15 -> 0..14), same convention as MARIDA's own
            # dataloader.py. No pixel is thrown away as ignore_index here.
            mask = mask - 1
        else:
            mask = remap_mados_mask(mask)

        img = np.moveaxis(img, 0, -1)  # (H, W, 11)

        n_glcm_bands = 0
        if self.use_glcm_texture:
            glcm_path = os.path.join(self.mados_path, scene_id, f'{scene_id}_L2R_glcm_{crop_id}.tif')
            if not os.path.exists(glcm_path):
                raise FileNotFoundError(
                    f"use_glcm_texture=True but no precomputed GLCM file found at {glcm_path}. "
                    f"Run precompute_mados_glcm.py --mados_path {self.mados_path} first."
                )
            ds = self._gdal.Open(glcm_path)
            glcm = ds.ReadAsArray().astype(np.float32)  # (6, H, W)
            ds = None
            glcm = np.moveaxis(glcm, 0, -1)  # (H, W, 6)
            n_glcm_bands = glcm.shape[-1]
            # Concatenate onto the raw bands BEFORE resize_to_256, so both
            # get padded/cropped identically and stay spatially aligned --
            # the GLCM file was computed on the same native (240x240)
            # resolution as the raw rhorc file, not the 256x256 model input.
            img = np.concatenate([img, glcm], axis=-1)  # (H, W, 11+6)

        img, mask = resize_to_256(img, mask)

        nan_mask = np.isnan(img)
        band_means = np.tile(BANDS_MEAN, (img.shape[0], img.shape[1], 1))
        if n_glcm_bands:
            # BANDS_MEAN only covers the 11 raw bands -- pad with zeros for
            # the GLCM channels' NaN-fill value (should be rare; GLCM
            # properties are all well-defined in [0,1]-ish ranges with no
            # natural NaNs from compute_glcm_features, this is a safety net).
            band_means = np.concatenate([band_means, np.zeros(img.shape[:2] + (n_glcm_bands,), dtype=np.float32)], axis=-1)
        img[nan_mask] = band_means[nan_mask]

        raw_img = img[..., :self.n_raw_bands]
        glcm_img = img[..., self.n_raw_bands:self.n_raw_bands + n_glcm_bands] if n_glcm_bands else None

        if self.use_spectral_indices or self.use_texture_features or self.use_glcm_texture:
            parts = []
            if self.use_spectral_indices:
                parts.append(compute_spectral_indices(raw_img))
            if self.use_glcm_texture:
                parts.append(glcm_img)
            elif self.use_texture_features:
                parts.append(compute_texture_features(raw_img))
            img = np.concatenate([raw_img] + parts, axis=-1)
        else:
            img = raw_img

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

        if self.use_spectral_indices or self.use_texture_features or self.use_glcm_texture:
            raw = img_t[:self.n_raw_bands]
            extra = img_t[self.n_raw_bands:]
            if self.spectral_jitter_prob > 0 and random.random() < self.spectral_jitter_prob:
                raw = self._spectral_jitter(raw)
            if self.standardization is not None:
                raw = self.standardization(raw)
            img_t = torch.cat([raw, extra], dim=0)
        else:
            if self.spectral_jitter_prob > 0 and random.random() < self.spectral_jitter_prob:
                img_t = self._spectral_jitter(img_t)
            if self.standardization is not None:
                img_t = self.standardization(img_t)

        return img_t, mask