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
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from utils.config import Config
from mssd_net import build_mssd_net
from dataloader import (
    GenDEBRIS, BANDS_MEAN, BANDS_STD, SPECTRAL_INDEX_NAMES, TEXTURE_FEATURE_NAMES,
)
from mados_dataloader import (
    MADOSDataset, MARIDA_LABELS, MARIDA_CLASSES_NOT_IN_MADOS,
)
from utils.metrics import Evaluation

# Same 6 property names as precompute_mados_glcm.py's GLCM_PROPERTIES --
# defined directly here (not imported) so this script only needs
# mados_dataloader.py to run --use_glcm_texture evaluation; the actual
# GLCM computation script is only needed once, to generate the cached
# feature files themselves, not at import time here.
GLCM_FEATURE_NAMES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

try:
    from utils.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

from evaluation_swin_unetv2 import tta_predict, per_class_report, format_report

os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)


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
    transform_deterministic = transforms.Compose([transforms.ToTensor()])

    if is_mados:
        dataset = MADOSDataset(
            options['mados_path'], split=options['mados_split'], transform=transform_deterministic,
            use_spectral_indices=options['use_spectral_indices'],
            use_texture_features=options['use_texture_features'],
            use_glcm_texture=options['use_glcm_texture'],
            standardization=standardization, splits_path=options['mados_splits_path'],
        )
        print(f"Loaded {len(dataset)} MADOS patches (split={options['mados_split']}).")
    else:
        transform_test = transform_deterministic
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
                         help="Path to the STACKED MADOS data (the '<something>_nearest' "
                              "folder produced by stack_patches.py). Required when --dataset mados.")
    parser.add_argument('--mados_split', default='test',
                         help="MADOS split name under --mados_splits_path -- resolves to "
                              "'<name>_X.txt' (e.g. 'test' -> test_X.txt), confirmed to match "
                              "MARIDA's own splits/ naming convention.")
    parser.add_argument('--mados_splits_path', default=None,
                         help="Path to the folder containing MADOS's {train,val,test}_X.txt. "
                              "Defaults to '<mados_path>/splits' if not given, but splits/ is "
                              "normally only created in the ORIGINAL (unstacked) MADOS/ folder, "
                              "not the stacked MADOS_nearest/ folder -- usually needs to be set "
                              "explicitly, e.g. --mados_splits_path /path/to/MADOS/splits.")
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
    parser.add_argument('--use_texture_features', default=False, type=bool,
                         help="Fast local std-dev + gradient magnitude proxy. Ignored if "
                              "--use_glcm_texture is also True.")
    parser.add_argument('--use_glcm_texture', default=False, type=bool,
                         help="Must match the --use_glcm_texture setting the checkpoint was "
                              "trained with. Requires precompute_mados_glcm.py to have been "
                              "run against --mados_path. Only relevant for --dataset mados.")
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

    # IMPORTANT: mirrors MADOSDataset's own internal priority logic exactly
    # -- if both use_texture_features and use_glcm_texture are True, GLCM
    # wins and the fast proxy is skipped (not stacked). Only relevant for
    # --dataset mados; --dataset marida never sets use_glcm_texture.
    effective_use_texture_features = options['use_texture_features'] and not options.get('use_glcm_texture', False)
    n_extra = (len(SPECTRAL_INDEX_NAMES) if options['use_spectral_indices'] else 0) \
        + (len(GLCM_FEATURE_NAMES) if options.get('use_glcm_texture', False) else
           (len(TEXTURE_FEATURE_NAMES) if effective_use_texture_features else 0))
    options['input_channels'] = 11 + n_extra

    main(options)