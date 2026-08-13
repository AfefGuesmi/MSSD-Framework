# -*- coding: utf-8 -*-
"""
ensemble_evaluate.py

Evaluate an ENSEMBLE of already-trained MSSD-Net checkpoints on the
MARIDA test split, by averaging their softmax predictions before argmax
(optionally combined with TTA per model). This costs zero retraining --
it's the free thing to try before any further training, since your own
per-class results already show the variants have complementary
strengths (e.g. `baseline` wins Sargassum classes, `only_residual` wins
Marine Debris and Foam under CE+Dice).

Reuses the exact same metric/report/logging pipeline as
evaluation_swin_unetv2.py, so the numbers are directly comparable to
your existing single-model logs.

USAGE (equal-weight ensemble of your three CE+Dice checkpoints):
    python ensemble_evaluate.py \
        --variants baseline only_residual full \
        --model_paths trained_models/swinunet-final/CE_dice/swin_unetv2_baseline_pretrained.pth \
                       trained_models/swinunet-final/CE_dice/swin_unetv2_only_residual_pretrained.pth \
                       trained_models/swinunet-final/CE_dice/swin_unetv2_full_pretrained.pth \
        --use_spectral_indices True --use_texture_features True

Weighted ensemble (e.g. trust only_residual and baseline more than full,
given full's weaker overall F1):
    ... --weights 0.4 0.4 0.2

--tta applies TTA independently to EACH model before the models'
predictions are averaged together (double-averaging: over augmentations,
then over models) -- this is more expensive but usually the strongest
option, since it combines both free-lunch tricks at once.
"""

import ast
import json
import logging
import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from utils.config import Config
from mssd_net import build_mssd_net, VARIANTS
from dataloader import GenDEBRIS, BANDS_MEAN, BANDS_STD, SPECTRAL_INDEX_NAMES, TEXTURE_FEATURE_NAMES
from utils.metrics import Evaluation

try:
    from utils.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

try:
    from utils.assets import labels as ALL_LABELS
except ImportError:
    ALL_LABELS = [
        'Marine Debris', 'Dense Sargassum', 'Sparse Sargassum',
        'Natural Organic Material', 'Ship', 'Clouds', 'Marine Water',
        'Sediment-Laden Water', 'Foam', 'Turbid Water', 'Shallow Water',
        'Waves', 'Cloud Shadows', 'Wakes', 'Mixed Water',
    ]

# Reuse the exact same TTA implementation as evaluation_swin_unetv2.py,
# so a --tta ensemble run is numerically consistent with your existing
# single-model TTA logs.
from evaluation_swin_unetv2 import tta_predict, per_class_report, format_report, HIGHLIGHT_CLASSES

os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)


def build_ensemble_models(variants, model_paths, options, device):
    """
    Construct and load one MSSDNet per (variant, checkpoint path) pair.
    Every model must share the same input channel count (i.e. all
    trained with the same --use_spectral_indices/--use_texture_features
    setting), since their predictions are combined in a single batch.
    """
    config = Config()
    swin_cfg = getattr(getattr(config, 'MODEL', None), 'SWIN', None)

    models = []
    for variant, path in zip(variants, model_paths):
        model = build_mssd_net(
            variant,
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
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if (
            isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
        ) else checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
        logging.info("Loaded %s checkpoint: %s", variant, path)
        print(f"Loaded {variant} checkpoint: {path}")

    return models


def ensemble_predict(models, weights, images, num_classes, options):
    """
    Average softmax probabilities across all models (each optionally
    TTA'd independently first), weighted by `weights`.
    """
    probs_sum = None
    for model, w in zip(models, weights):
        if options['tta']:
            probs = tta_predict(model, images, num_classes,
                                 rotations=options['tta_rotations'],
                                 use_hflip=options['tta_hflip'])
        else:
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=1)
        probs = probs * w
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum  # weights already applied; caller sums to argmax directly


def main(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = options['variants']
    model_paths = options['model_paths']
    if len(variants) != len(model_paths):
        raise ValueError(f"Got {len(variants)} --variants but {len(model_paths)} --model_paths -- "
                          "these must be given in matching order, one path per variant.")

    weights = options['weights']
    if weights is None:
        weights = [1.0 / len(variants)] * len(variants)
    else:
        if len(weights) != len(variants):
            raise ValueError(f"Got {len(weights)} --weights but {len(variants)} models -- "
                              "must supply exactly one weight per model, or omit --weights "
                              "for an equal-weight ensemble.")
        total = sum(weights)
        weights = [w / total for w in weights]  # normalize so they sum to 1

    tag = 'ensemble_' + '_'.join(variants)
    log_name = f"evaluating_swin_{tag}"
    if options['tta']:
        log_name += '_tta'
    log_path = os.path.join(PROJECT_ROOT, 'logs', f'{log_name}.log')
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    description = f"Ensemble of {list(zip(variants, [round(w, 3) for w in weights]))}"
    logging.info('*' * 10)
    logging.info('Evaluating: %s%s', description, ' [TTA per model: 4 rotations x hflip]' if options['tta'] else '')
    print(f"Evaluating: {description}")

    models = build_ensemble_models(variants, model_paths, options, device)

    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)

    dataset_test = GenDEBRIS('test', transform=transform_test,
                              standardization=standardization,
                              agg_to_water=options['agg_to_water'],
                              use_spectral_indices=options['use_spectral_indices'],
                              use_texture_features=options['use_texture_features'],
                              splits_dir=options['splits_dir'])
    test_loader = DataLoader(dataset_test, batch_size=options['batch'], shuffle=False)

    class_names = list(ALL_LABELS)
    if options['agg_to_water']:
        class_names = class_names[:-4]

    y_true = []
    y_predicted = []

    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc="testing (ensemble)"):
            image = image.to(device)
            target = target.to(device)

            probs = ensemble_predict(models, weights, image, options['output_channels'], options)

            probs = torch.movedim(probs, 1, -1).reshape(-1, options['output_channels'])
            target = target.reshape(-1)

            mask = target != -1
            probs = probs[mask]
            target = target[mask]

            probs = probs.cpu().numpy()
            target = target.cpu().numpy()

            y_predicted += probs.argmax(1).tolist()
            y_true += target.tolist()

    acc = Evaluation(y_predicted, y_true)
    logging.info("\n")
    logging.info("STATISTICS: \n")
    logging.info("Evaluation: " + str(acc))
    print("Evaluation: " + str(acc))

    if confusion_matrix is not None:
        conf_mat = confusion_matrix(y_true, y_predicted, class_names)
        logging.info("Confusion Matrix:  \n" + str(conf_mat.to_string()))
        print("Confusion Matrix:  \n" + str(conf_mat.to_string()))

    report = per_class_report(y_true, y_predicted, class_names)
    report_text = format_report(report)

    logging.info("Per-class quantitative results (%s):\n%s", description, report_text)
    print(f"\nPer-class quantitative results ({description}):")
    print(report_text)

    json_path = os.path.join(PROJECT_ROOT, 'logs', f'{log_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'model_type': 'swin_ensemble', 'tag': tag,
            'description': description,
            'variants': variants, 'model_paths': model_paths, 'weights': weights,
            'per_class': report,
        }, f, indent=2)
    logging.info("Saved machine-readable results to: %s", json_path)
    print(f"Saved machine-readable results to: {json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variants', nargs='+', required=True,
                         choices=list(VARIANTS.keys()),
                         help="Space-separated list of MSSD-Net variants to ensemble, e.g. "
                              "'--variants baseline only_residual full'.")
    parser.add_argument('--model_paths', nargs='+', required=True,
                         help="Space-separated list of checkpoint paths, ONE PER VARIANT, "
                              "in the SAME ORDER as --variants.")
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                         help="Optional per-model weights (need not sum to 1 -- normalized "
                              "automatically), same order as --variants. Default: equal weight "
                              "for every model.")
    parser.add_argument('--agg_to_water', default=True, type=bool)
    parser.add_argument('--splits_dir', default='splits', type=str,
                         help="Must match the --splits_dir every checkpoint was trained with.")
    parser.add_argument('--batch', default=5, type=int)
    parser.add_argument('--input_channels', default=11, type=int,
                         help="Overridden automatically based on --use_spectral_indices/"
                              "--use_texture_features, same as evaluation_swin_unetv2.py. "
                              "ALL ensembled models must share this input shape.")
    parser.add_argument('--output_channels', default=11, type=int)
    parser.add_argument('--use_spectral_indices', default=False, type=bool,
                         help="Must match what ALL ensembled checkpoints were trained with.")
    parser.add_argument('--use_texture_features', default=False, type=bool,
                         help="Must match what ALL ensembled checkpoints were trained with.")
    parser.add_argument('--tta', default=True, type=bool,
                         help="Apply TTA independently to each model before averaging across "
                              "models (double-averaging: over augmentations, then over models). "
                              "More expensive, usually the strongest option.")
    parser.add_argument('--tta_rotations', default='[0,180]', type=str)
    parser.add_argument('--tta_hflip', default=False, type=bool)

    args = parser.parse_args()
    options = vars(args)

    options['tta_rotations'] = ast.literal_eval(options['tta_rotations'])
    if not isinstance(options['tta_rotations'], (list, tuple)):
        options['tta_rotations'] = [options['tta_rotations']]

    n_extra = (len(SPECTRAL_INDEX_NAMES) if options['use_spectral_indices'] else 0) \
        + (len(TEXTURE_FEATURE_NAMES) if options['use_texture_features'] else 0)
    options['input_channels'] = 11 + n_extra

    main(options)
