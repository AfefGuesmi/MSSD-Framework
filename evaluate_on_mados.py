# -*- coding: utf-8 -*-
"""
evaluate_on_mados.py

Evaluate an already-trained MARIDA checkpoint on EITHER the MARIDA test
set OR the MADOS dataset, selected via --dataset {marida,mados}, both
reported in MARIDA's 11-class label space so results are directly
comparable side by side using the exact same model, metrics, and report
formatting -- switching --dataset is the only thing that changes.

--dataset marida: identical to evaluation_swin_unetv2.py's test-set
    evaluation (same GenDEBRIS loader, same --splits_dir support). Lets
    you use this one script instead of maintaining two separate tools.

--dataset mados (Kikaki et al., 2024 -- built by the same team as
    MARIDA, as a successor/extension): a cross-dataset GENERALIZATION
    check, remapped onto MARIDA's 11-class label space. This is NOT
    directly comparable to MARIDA test-set numbers in the way two
    MARIDA runs are -- different data, different annotation protocol,
    and 4 MADOS classes (Oil Spills, Oil Platforms, Jellyfish, Sea Snot)
    have no MARIDA equivalent and are excluded from evaluation entirely.
    Report --dataset mados results as a separate generalization result,
    not merged into your main MARIDA results table.

======================================================================
VERIFIED vs. UNVERIFIED (MADOS-specific) -- read before trusting output
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
    confirmed via PANGAEA. This script center-crops/pads to 256x256 to
    match your model's expected input; adjust MADOS_PATCH_SIZE below if
    this is wrong for your actual download.

NOT VERIFIED -- please confirm on your actual download before trusting
results, same discipline as the confidence-raster naming issue earlier:
  - The exact file/folder naming convention for MADOS patches once
    downloaded and "stacked" (their own README describes a required
    `utils/stack_patches.py` step to combine raw per-band rasters into
    a single multiband GeoTIFF per patch -- this script assumes that
    step has already been run and produces `<name>.tif` / `<name>_cl.tif`
    pairs in the same style as MARIDA's dataloader, but this is an
    ASSUMPTION based on the MARIDA-derived codebase pattern, not a
    confirmed file listing).
  - Whether MADOS's raster mask values are 1-indexed the same way
    MARIDA's are (mask - 1 shift), or already 0-indexed.

IMPORTANT -- MADOS has NO "Clouds" class:
MARIDA's class 5 (Clouds) does not appear anywhere in MADOS's 15-class
list. This means Clouds will have zero ground-truth support when
evaluating on MADOS -- this is NOT a model failure, it's simply a class
MADOS doesn't test. With --dataset mados, this script explicitly
EXCLUDES Clouds from the macro-average (with --dataset marida, Clouds
is included as normal, since MARIDA does annotate it).

USAGE:
    # Evaluate on MARIDA's own test set (same as evaluation_swin_unetv2.py)
    python evaluate_on_mados.py --dataset marida \
        --variant baseline --use_spectral_indices True --use_texture_features True \
        --model_path trained_models/swinunet-final/CE_dice/swin_unetv2_baseline_pretrained.pth

    # Evaluate the SAME checkpoint on MADOS, for a generalization check
    python evaluate_on_mados.py --dataset mados \
        --variant baseline --use_spectral_indices True --use_texture_features True \
        --mados_path /path/to/MADOS \
        --model_path trained_models/swinunet-final/CE_dice/swin_unetv2_baseline_pretrained.pth
"""

import argparse
import ast
import json
import logging
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from utils.config import Config
from mssd_net import build_mssd_net
from dataloader import (
    GenDEBRIS, BANDS_MEAN, BANDS_STD, SPECTRAL_INDEX_NAMES, TEXTURE_FEATURE_NAMES,
    compute_spectral_indices, compute_texture_features,
)
from utils.metrics import Evaluation

try:
    from utils.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

from evaluation_swin_unetv2 import tta_predict, per_class_report, format_report

os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

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
# excluded from evaluation entirely rather than guessed-at).
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
#                                                            agg_to_water=True logic,
#                                                            which folds Waves/Wakes
#                                                            into Marine Water
#   Oil Platforms              -> None (ignored)              no MARIDA equivalent
#   Jellyfish Aggregations     -> None (ignored)              no MARIDA equivalent
#   Sea Snot                   -> None (ignored)              no MARIDA equivalent
MADOS_TO_MARIDA = {
    0: 0,    # Marine Debris -> Marine Debris
    1: None,  # Oil Spills -> ignored
    2: 1,    # Dense Sargassum -> Dense Sargassum
    3: 2,    # Sparse Floating Algae -> Sparse Sargassum
    4: 3,    # Natural Organic Material -> Natural Organic Material
    5: 4,    # Ships -> Ship
    6: 6,    # Marine Water -> Marine Water
    7: 7,    # Sediment-Laden Water -> Sediment-Laden Water
    8: 8,    # Foam -> Foam
    9: 9,    # Turbid Water -> Turbid Water
    10: 10,  # Shallow Water -> Shallow Water
    11: 6,   # Waves and Wakes -> Marine Water (matches MARIDA's own agg_to_water)
    12: None,  # Oil Platforms -> ignored
    13: None,  # Jellyfish Aggregations -> ignored
    14: None,  # Sea Snot -> ignored
}

# MARIDA classes that MADOS cannot test at all (no equivalent in its
# label set) -- excluded from the macro average computed here, reported
# separately instead of silently scored as 0%.
MARIDA_CLASSES_NOT_IN_MADOS = ['Clouds']


def remap_mados_mask(mados_mask):
    """
    Convert a raw MADOS mask (1-indexed class codes, 0/background/void as
    needed) into MARIDA's 0-indexed 11-class label space, using
    MADOS_TO_MARIDA. Unmapped classes and anything outside the valid
    MADOS range become -1 (ignore_index), matching how MARIDA's own
    masks already encode "don't evaluate this pixel".
    """
    remapped = np.full_like(mados_mask, -1)
    for mados_idx, marida_idx in MADOS_TO_MARIDA.items():
        if marida_idx is not None:
            remapped[mados_mask == (mados_idx + 1)] = marida_idx  # +1: MADOS masks are 1-indexed
    return remapped


class MADOSDataset(Dataset):
    """
    Minimal MADOS loader: reads already-stacked <name>.tif / <name>_cl.tif
    pairs (see the NOT VERIFIED note at the top of this file regarding
    this file-naming assumption), remaps labels to MARIDA's space via
    remap_mados_mask, optionally computes the same spectral-index/texture
    extra channels your model was trained with, and center-crops/pads to
    256x256 to match MARIDA's patch size.
    """

    def __init__(self, mados_path, split='test', use_spectral_indices=False,
                 use_texture_features=False, standardization=None):
        from osgeo import gdal

        self._gdal = gdal
        self.mados_path = mados_path
        self.use_spectral_indices = use_spectral_indices
        self.use_texture_features = use_texture_features
        self.standardization = standardization
        self.n_raw_bands = len(BANDS_MEAN)

        split_file = os.path.join(mados_path, 'splits', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Could not find {split_file}. This script assumes a MADOS split file layout "
                f"similar to MARIDA's -- check your actual MADOS download structure and adjust "
                f"this path (and the patch-file naming below) to match."
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

        # Center pad/crop 240x240 -> 256x256 to match MARIDA's patch size.
        img = np.moveaxis(img, 0, -1)  # (H, W, C)
        img, mask = self._resize_to_256(img, mask)

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

        img = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
        mask = torch.from_numpy(mask)

        if self.use_spectral_indices or self.use_texture_features:
            raw = img[:self.n_raw_bands]
            extra = img[self.n_raw_bands:]
            if self.standardization is not None:
                raw = self.standardization(raw)
            img = torch.cat([raw, extra], dim=0)
        elif self.standardization is not None:
            img = self.standardization(img)

        return img, mask

    @staticmethod
    def _resize_to_256(img, mask, target=256):
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


def build_model(options, device):
    config = Config()
    swin_cfg = getattr(getattr(config, 'MODEL', None), 'SWIN', None)
    model = build_mssd_net(
        options['variant'],
        img_size=config.DATA.IMG_SIZE,
        patch_size=getattr(swin_cfg, 'PATCH_SIZE', 4),
        in_chans=options['input_channels'],
        num_classes=options['output_channels'],
        embed_dim=getattr(swin_cfg, 'EMBED_DIM', 96),
        depths=getattr(swin_cfg, 'DEPTHS', [2, 2, 2, 2]),
        depths_decoder=getattr(swin_cfg, 'DEPTHS_DECODER', [1, 2, 2, 2]),
        num_heads=getattr(swin_cfg, 'NUM_HEADS', [3, 6, 12, 24]),
        window_size=getattr(swin_cfg, 'WINDOW_SIZE', 8),
    )
    model.to(device)
    return model


def main(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_mados = options['dataset'] == 'mados'

    log_name = f"evaluating_on_{options['dataset']}_{options['variant']}"
    if options['tta']:
        log_name += '_tta'
    log_path = os.path.join(PROJECT_ROOT, 'logs', f'{log_name}.log')
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    logging.info('*' * 10)
    logging.info('Evaluating on %s, checkpoint variant=%s%s',
                  options['dataset'].upper(), options['variant'], ' [TTA]' if options['tta'] else '')
    if is_mados:
        logging.info('NOTE: Clouds excluded from macro average -- MADOS has no Clouds annotations.')
        print('Cross-dataset evaluation on MADOS (see log for important caveats about class coverage).')
    else:
        print('Evaluation on MARIDA test set.')

    model = build_model(options, device)
    checkpoint = torch.load(options['model_path'], map_location=device)
    state_dict = checkpoint['model_state_dict'] if (
        isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
    ) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)

    if is_mados:
        dataset = MADOSDataset(
            options['mados_path'], split=options['mados_split'],
            use_spectral_indices=options['use_spectral_indices'],
            use_texture_features=options['use_texture_features'],
            standardization=standardization,
        )
        print(f"Loaded {len(dataset)} MADOS patches (split={options['mados_split']}).")
    else:
        transform_test = transforms.Compose([transforms.ToTensor()])
        dataset = GenDEBRIS(
            'test', transform=transform_test, standardization=standardization,
            agg_to_water=options['agg_to_water'],
            use_spectral_indices=options['use_spectral_indices'],
            use_texture_features=options['use_texture_features'],
            splits_dir=options['splits_dir'],
        )
        print(f"Loaded {len(dataset)} MARIDA test patches (splits_dir={options['splits_dir']}).")

    loader = DataLoader(dataset, batch_size=options['batch'], shuffle=False)

    y_true, y_predicted = [], []
    with torch.no_grad():
        for image, target in tqdm(loader, desc=f"testing ({options['dataset']})"):
            image = image.to(device)
            target = target.to(device)

            if options['tta']:
                probs = tta_predict(model, image, options['output_channels'],
                                     rotations=options['tta_rotations'], use_hflip=options['tta_hflip'])
            else:
                probs = torch.nn.functional.softmax(model(image), dim=1)

            probs = torch.movedim(probs, 1, -1).reshape(-1, options['output_channels'])
            target = target.reshape(-1)
            mask = target != -1
            probs, target = probs[mask], target[mask]

            y_predicted += probs.cpu().numpy().argmax(1).tolist()
            y_true += target.cpu().numpy().tolist()

    if not y_true:
        raise RuntimeError(f"No valid (non-ignored) pixels found for --dataset {options['dataset']} -- "
                            f"check the class crosswalk and file paths before assuming this ran correctly.")

    acc = Evaluation(y_predicted, y_true)
    logging.info("Evaluation: %s", acc)
    if is_mados:
        print("Evaluation (includes Clouds at support=0 -- see note above):", acc)
    else:
        print("Evaluation:", acc)

    if confusion_matrix is not None:
        conf_mat = confusion_matrix(y_true, y_predicted, MARIDA_LABELS)
        logging.info("Confusion Matrix:\n%s", conf_mat.to_string())

    report = per_class_report(y_true, y_predicted, MARIDA_LABELS)
    report_text = format_report(report)
    logging.info("Per-class results:\n%s", report_text)
    print(report_text)

    result_payload = {
        'dataset': options['dataset'],
        'variant': options['variant'], 'model_path': options['model_path'],
        'per_class': report,
    }

    if is_mados:
        # Macro average EXCLUDING Clouds (the honest number for MADOS,
        # since Clouds has no MADOS annotations at all -- see caveats).
        testable_classes = [c for c in MARIDA_LABELS if c not in MARIDA_CLASSES_NOT_IN_MADOS]
        macro_f1_no_clouds = np.mean([report[c]['f1'] for c in testable_classes])
        macro_iou_no_clouds = np.mean([report[c]['iou'] for c in testable_classes])
        print(f"\nMacro F1 excluding Clouds (not annotated in MADOS): {macro_f1_no_clouds*100:.2f}%")
        print(f"Macro mIoU excluding Clouds: {macro_iou_no_clouds*100:.2f}%")
        logging.info("Macro F1 excluding Clouds: %.4f, mIoU excluding Clouds: %.4f",
                      macro_f1_no_clouds, macro_iou_no_clouds)
        result_payload['macro_f1_excluding_clouds'] = float(macro_f1_no_clouds)
        result_payload['macro_iou_excluding_clouds'] = float(macro_iou_no_clouds)
        result_payload['note'] = 'Clouds excluded from macro average -- MADOS has no Clouds annotations.'

    json_path = os.path.join(PROJECT_ROOT, 'logs', f'{log_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump(result_payload, f, indent=2)
    print(f"Saved results to: {json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='mados', choices=['marida', 'mados'],
                         help="Which dataset to evaluate on. 'marida' = MARIDA's own test set "
                              "(same as evaluation_swin_unetv2.py). 'mados' = cross-dataset "
                              "generalization check on MADOS, remapped to MARIDA's label space.")
    parser.add_argument('--mados_path', default=None,
                         help='Root MADOS directory. Required when --dataset mados.')
    parser.add_argument('--mados_split', default='test', help="MADOS split file name under <mados_path>/splits/")
    parser.add_argument('--splits_dir', default='splits', type=str,
                         help="MARIDA splits subfolder. Only used when --dataset marida -- "
                              "must match whatever the checkpoint was trained/evaluated with "
                              "(e.g. 'splits_80_10_10' for the custom stratified split).")
    parser.add_argument('--agg_to_water', default=True, type=bool,
                         help='Only used when --dataset marida.')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--batch', default=5, type=int)
    parser.add_argument('--output_channels', default=11, type=int)
    parser.add_argument('--input_channels', default=11, type=int)
    parser.add_argument('--use_spectral_indices', default=False, type=bool)
    parser.add_argument('--use_texture_features', default=False, type=bool)
    parser.add_argument('--tta', default=True, type=bool)
    parser.add_argument('--tta_rotations', default='[0,180]', type=str)
    parser.add_argument('--tta_hflip', default=False, type=bool)
    args = parser.parse_args()
    options = vars(args)

    if options['dataset'] == 'mados' and not options['mados_path']:
        parser.error("--mados_path is required when --dataset mados")

    options['tta_rotations'] = ast.literal_eval(options['tta_rotations'])
    if not isinstance(options['tta_rotations'], (list, tuple)):
        options['tta_rotations'] = [options['tta_rotations']]

    n_extra = (len(SPECTRAL_INDEX_NAMES) if options['use_spectral_indices'] else 0) \
        + (len(TEXTURE_FEATURE_NAMES) if options['use_texture_features'] else 0)
    options['input_channels'] = 11 + n_extra

    main(options)
