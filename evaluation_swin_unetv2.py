# -*- coding: utf-8 -*-
"""
evaluation_swin_unetv2.py

Quantitative evaluation for BOTH models trained in this project:
  * MSSD-Net (Swin-UNet V2 encoder/decoder plus the dilated-bottleneck /
    attention-refinement / residual-fusion ablation modules), via
    --model_type swin --variant <baseline|+dilated|+dilated+attention|
    full|only_dilated|only_attention|only_residual>
  * The U-Net baseline (plain, or with a resnet18/mobilenetv2/
    efficientnetv2 backbone), via --model_type unet --backbone <...>

For a given trained checkpoint this computes, on the MARIDA test split:
  * the project-standard overall Evaluation(...) + confusion_matrix(...)
    (unchanged, so numbers stay directly comparable across every model
    and variant evaluated with this script)
  * a per-class Precision / Recall / F1 / IoU breakdown for every class,
    with Marine Debris, Dense Sargassum and Sparse Sargassum called out
    explicitly since they're the classes of actual interest, plus the
    macro-average across all classes

Everything is printed to stdout AND appended to a log file, plus dumped
as JSON, both named after the model type + variant/backbone being
evaluated (e.g. evaluating_swin_full.log, evaluating_unet_mobilenetv2.log)
so evaluating multiple models/variants back-to-back never overwrites a
previous run's results.

Optionally also writes full-resolution prediction masks to disk, as the
original evaluation.py did for the plain UNet.
"""

import os
import sys
import json
import ast
import random
import logging
import argparse
import numpy as np
import rasterio
from tqdm import tqdm
from os.path import dirname as up

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

PROJECT_ROOT = up(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from utils.config import Config
from mssd_net import build_mssd_net, VARIANTS
from unet import UNet
from dataloader import GenDEBRIS, BANDS_MEAN, BANDS_STD, DATASET_PATH
from utils.metrics import Evaluation

try:
    from utils.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

try:
    from utils.assets import labels as ALL_LABELS
except ImportError:
    # Fallback: standard MARIDA class order, used only if utils/assets.py
    # isn't found under this project layout.
    ALL_LABELS = [
        'Marine Debris', 'Dense Sargassum', 'Sparse Sargassum',
        'Natural Organic Material', 'Ship', 'Clouds', 'Marine Water',
        'Sediment-Laden Water', 'Foam', 'Turbid Water', 'Shallow Water',
        'Waves', 'Cloud Shadows', 'Wakes', 'Mixed Water',
    ]

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

# Classes always called out by name in the report, on top of the overall
# macro-average -- these are the classes that actually matter for marine
# debris detection, regardless of where they happen to fall in the full
# label list or how many total classes there are after aggregation.
HIGHLIGHT_CLASSES = ['Marine Debris', 'Dense Sargassum', 'Sparse Sargassum']


def tta_predict(model, images, num_classes, rotations=(0, 180), use_hflip=False):
    """
    Test-time augmentation: average softmax probabilities over every
    combination of the given 90-degree-multiple rotations and an optional
    horizontal flip. Defaults to a cheap 2-pass 0/180 rotation-only sweep;
    pass rotations=(0,90,180,270), use_hflip=True for the fuller 8-pass
    version matching the training augmentation space (RandomRotation
    Transform([-90, 0, 90, 180]) + RandomHorizontalFlip in
    train_swin_unetv2.py), just enumerated exhaustively instead of sampled.
    Each augmented prediction is un-rotated/un-flipped back to the original
    orientation before averaging, so probabilities stay pixel-aligned with
    the (un-augmented) target mask.

    Args:
        model: the segmentation model, called as model(images) -> (B, C, H, W) logits.
        images (torch.Tensor): (B, C_in, H, W) input batch, already standardized.
        num_classes (int): number of output classes (unused directly, kept
            for a clear call signature / easy assertion if needed).
        rotations (tuple[int]): rotation angles in degrees, each must be a
            multiple of 90 (0/90/180/270). Fewer angles = cheaper TTA.
        use_hflip (bool): if True, also average over a horizontal flip at
            each rotation (doubles the pass count); if False, rotations only.

    Returns:
        torch.Tensor: (B, num_classes, H, W) averaged softmax probabilities.
    """
    for angle in rotations:
        if angle % 90 != 0:
            raise ValueError(f"tta_predict only supports 90-degree-multiple rotations, got {angle}")

    flips = (False, True) if use_hflip else (False,)
    probs_sum = None
    n_passes = 0

    for angle in rotations:
        k = (angle // 90) % 4  # torch.rot90 turns count, 0..3
        for flip in flips:
            aug = torch.rot90(images, k=k, dims=(-2, -1))
            if flip:
                aug = torch.flip(aug, dims=(-1,))

            logits = model(aug)  # (B, num_classes, H, W)

            if flip:
                logits = torch.flip(logits, dims=(-1,))
            logits = torch.rot90(logits, k=-k, dims=(-2, -1))

            probs = torch.nn.functional.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            n_passes += 1

    return probs_sum / n_passes


def find_latest_checkpoint(model_dir, checkpoint_name='best_model.pth'):
    """
    Fall back to the highest-numbered epoch folder under a checkpoint
    directory if --model_path isn't given explicitly (mirrors how both
    train_swin_unetv2.py and train.py name checkpoint folders by epoch).
    """
    if not os.path.isdir(model_dir):
        return None
    epoch_dirs = [d for d in os.listdir(model_dir) if d.isdigit()]
    if not epoch_dirs:
        return None
    latest = max(epoch_dirs, key=int)
    candidate = os.path.join(model_dir, latest, checkpoint_name)
    return candidate if os.path.isfile(candidate) else None


def build_model(options, device):
    """
    Construct either MSSDNet (Swin-UNet V2 + ablation modules) or the
    U-Net baseline, matching whichever training script produced the
    checkpoint being evaluated.

    Returns (model, tag, description) where `tag` identifies this run
    for log/JSON/mask filenames (the variant name for swin, the backbone
    name for unet) and `description` is a short human-readable summary
    logged at the start of evaluation.
    """
    if options['model_type'] == 'swin':
        variant = options['variant']
        config = Config()
        swin_cfg = getattr(config, 'MODEL', None)
        swin_cfg = getattr(swin_cfg, 'SWIN', None)

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
        tag = variant
        description = f"MSSDNet variant={variant} ({VARIANTS[variant]})"
    else:
        backbone = options['backbone']
        model = UNet(
            input_bands=options['input_channels'],
            output_classes=options['output_channels'],
            hidden_channels=options['hidden_channels'],
            backbone=backbone,
        )
        tag = backbone
        description = f"UNet backbone={backbone}, hidden_channels={options['hidden_channels']}"

    model.to(device)
    return model, tag, description


def resolve_checkpoint(options, tag):
    """
    Find the checkpoint to load: an explicit --model_path always wins.
    Otherwise, look under trained_models/<tag>/ first (how
    train_swin_unetv2.py always namespaces checkpoints, and how you may
    have organised U-Net runs by backbone yourself), then fall back to
    trained_models/ directly (train.py's un-namespaced default location).
    """
    if options['model_path']:
        return options['model_path']

    tagged_dir = os.path.join(PROJECT_ROOT, 'trained_models', tag)
    model_file = find_latest_checkpoint(tagged_dir, options['checkpoint_name'])
    if model_file is not None:
        return model_file

    base_dir = os.path.join(PROJECT_ROOT, 'trained_models')
    model_file = find_latest_checkpoint(base_dir, options['checkpoint_name'])
    if model_file is not None:
        return model_file

    raise FileNotFoundError(
        f"No checkpoint found under {tagged_dir} or {base_dir}. "
        f"Pass --model_path explicitly."
    )


def per_class_report(y_true, y_pred, class_names):
    """
    Vectorised Precision / Recall / F1 / IoU per class (pure numpy, no
    sklearn dependency), plus the macro-average across all classes.

    Returns a dict, ordered by class_names, with an 'Average' entry
    appended last holding the macro-average.
    """
    num_classes = len(class_names)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    idx = y_true * num_classes + y_pred
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.nan_to_num(tp / (tp + fp), nan=0.0)
        recall = np.nan_to_num(tp / (tp + fn), nan=0.0)
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall), nan=0.0)
        iou = np.nan_to_num(tp / (tp + fp + fn), nan=0.0)

    report = {}
    for i, name in enumerate(class_names):
        report[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'iou': float(iou[i]),
            'support': int(support[i]),
        }

    report['Average'] = {
        'precision': float(precision.mean()),
        'recall': float(recall.mean()),
        'f1': float(f1.mean()),
        'iou': float(iou.mean()),
        'support': int(support.sum()),
    }
    return report


def format_report(report, highlight=HIGHLIGHT_CLASSES):
    """Aligned text table: highlighted classes first, then every other
    class, then the macro-average last."""
    header = '{:25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        'Class', 'Precision', 'Recall', 'F1', 'IoU', 'Support')
    sep = '-' * len(header)
    lines = [header, sep]

    ordered_names = [n for n in highlight if n in report]
    ordered_names += [n for n in report if n not in ordered_names and n != 'Average']

    for name in ordered_names:
        m = report[name]
        lines.append('{:25s} {:9.2f}% {:9.2f}% {:9.2f}% {:9.2f}% {:10d}'.format(
            name, m['precision'] * 100, m['recall'] * 100, m['f1'] * 100,
            m['iou'] * 100, m['support']))

    lines.append(sep)
    m = report['Average']
    lines.append('{:25s} {:9.2f}% {:9.2f}% {:9.2f}% {:9.2f}% {:10d}'.format(
        'Average (macro)', m['precision'] * 100, m['recall'] * 100, m['f1'] * 100,
        m['iou'] * 100, m['support']))
    return '\n'.join(lines)


def main(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tag, description = build_model(options, device)

    # Per-model/variant log file so evaluating swin (any variant) and
    # unet (any backbone) back-to-back never clobber each other's results.
    # TTA runs get their own '_tta' suffix so they don't overwrite the
    # non-TTA results for the same checkpoint.
    log_name = f"evaluating_{options['model_type']}_{tag}"
    if options['tta']:
        log_name += '_tta'
    log_path = os.path.join(PROJECT_ROOT, 'logs', f'{log_name}.log')
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    logging.info('*' * 10)
    logging.info('Evaluating: %s%s', description, ' [TTA: 4 rotations x hflip]' if options['tta'] else '')

    # Transformations
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)

    # Data
    dataset_test = GenDEBRIS('test', transform=transform_test,
                              standardization=standardization,
                              agg_to_water=options['agg_to_water'])
    test_loader = DataLoader(dataset_test, batch_size=options['batch'], shuffle=False)

    class_names = list(ALL_LABELS)
    if options['agg_to_water']:
        class_names = class_names[:-4]  # drop Mixed Water, Wakes, Cloud Shadows, Waves

    # Resolve checkpoint path (explicit --model_path wins; otherwise
    # auto-locate the latest epoch under trained_models/<tag>/, falling
    # back to trained_models/ directly for un-namespaced U-Net runs).
    model_file = resolve_checkpoint(options, tag)
    logging.info('Loading model checkpoint: %s', model_file)
    print(f"Loading model checkpoint: {model_file}")

    checkpoint = torch.load(model_file, map_location=device)
    state_dict = checkpoint['model_state_dict'] if (
        isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
    ) else checkpoint
    model.load_state_dict(state_dict)

    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()


    y_true = []
    y_predicted = []

    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc="testing"):

            image = image.to(device)
            target = target.to(device)

            if options['tta']:
                probs = tta_predict(model, image, options['output_channels'],
                                     rotations=options['tta_rotations'],
                                     use_hflip=options['tta_hflip'])  # (B, num_classes, H, W)
            else:
                logits = model(image)  # (B, num_classes, H, W)
                probs = torch.nn.functional.softmax(logits, dim=1)

            # -> (B, H, W, num_classes) -> (-1, num_classes), matching the
            # flattened per-pixel target below. Accuracy only computed on
            # annotated pixels (target != -1).
            probs = torch.movedim(probs, 1, -1).reshape(-1, options['output_channels'])
            target = target.reshape(-1)

            mask = target != -1
            probs = probs[mask]
            target = target[mask]

            probs = probs.cpu().numpy()
            target = target.cpu().numpy()

            y_predicted += probs.argmax(1).tolist()
            y_true += target.tolist()

    ########################################################################
    # Overall project-standard metrics (unchanged from the UNet baseline
    # script, so results stay directly comparable across experiments).
    ########################################################################
    acc = Evaluation(y_predicted, y_true)
    logging.info("\n")
    logging.info("STATISTICS: \n")
    logging.info("Evaluation: " + str(acc))
    print("Evaluation: " + str(acc))

    if confusion_matrix is not None:
        conf_mat = confusion_matrix(y_true, y_predicted, class_names)
        logging.info("Confusion Matrix:  \n" + str(conf_mat.to_string()))
        print("Confusion Matrix:  \n" + str(conf_mat.to_string()))

    ########################################################################
    # Per-class quantitative results: macro-average across all classes,
    # plus Marine Debris / Dense Sargassum / Sparse Sargassum called out
    # by name, printed AND logged.
    ########################################################################
    report = per_class_report(y_true, y_predicted, class_names)
    report_text = format_report(report)

    logging.info("Per-class quantitative results (%s):\n%s", description, report_text)
    print(f"\nPer-class quantitative results ({description}):")
    print(report_text)

    # Machine-readable copy, also named by model type + tag, for later
    # comparison across models and ablation variants.
    json_path = os.path.join(PROJECT_ROOT, 'logs', f'{log_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'model_type': options['model_type'], 'tag': tag,
            'description': description, 'model_path': model_file,
            'per_class': report,
        }, f, indent=2)
    logging.info("Saved machine-readable results to: %s", json_path)
    print(f"Saved machine-readable results to: {json_path}")

    ########################################################################
    # Optional: full-resolution prediction masks (unchanged in behaviour
    # from the original UNet evaluation.py, named by model type + tag so
    # different models/experiments don't overwrite each other's masks).
    ########################################################################
    if options['predict_masks']:

        path = os.path.join(DATASET_PATH, 'patches')
        ROIs = np.genfromtxt(os.path.join(DATASET_PATH, 'splits', 'test_X.txt'), dtype='str')

        impute_nan = np.tile(BANDS_MEAN, (256, 256, 1))

        os.makedirs(options['gen_masks_path'], exist_ok=True)

        for roi in tqdm(ROIs, desc="saving masks"):

            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])
            roi_name = '_'.join(['S2'] + roi.split('_'))
            roi_file = os.path.join(path, roi_folder, roi_name + '.tif')

            output_image = os.path.join(
                options['gen_masks_path'],
                os.path.basename(roi_file).split('.tif')[0] + f'_{log_name}.tif'
            )

            with rasterio.open(roi_file, mode='r') as src:
                tags = src.tags().copy()
                meta = src.meta
                image = src.read()
                image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
                dtype = src.read(1).dtype

            meta.update(count=1)

            with rasterio.open(output_image, 'w', **meta) as dst:

                nan_mask = np.isnan(image)
                image[nan_mask] = impute_nan[nan_mask]

                image = transform_test(image)
                image = standardization(image)
                image = image.to(device)

                if options['tta']:
                    probs = tta_predict(model, image.unsqueeze(0), options['output_channels'],
                                        rotations=options['tta_rotations'],
                                        use_hflip=options['tta_hflip'])
                else:
                    logits = model(image.unsqueeze(0))
                    probs = torch.nn.functional.softmax(logits.detach(), dim=1)
                probs = probs.detach().cpu().numpy()
                probs = probs.argmax(1).squeeze() + 1

                dst.write_band(1, probs.astype(dtype).copy())
                dst.update_tags(**tags)

    return report


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='swin', choices=['swin', 'unet'],
                         help='Which model architecture to evaluate: the MSSD-Net '
                              '(Swin-UNet V2 + ablation modules) or the U-Net baseline')

    # --- Swin/MSSD-Net specific ---
    parser.add_argument('--variant', default='full', choices=list(VARIANTS.keys()),
                         help='[--model_type swin] Which MSSD-Net ablation variant to '
                              'evaluate -- must match the architecture used to train '
                              'the checkpoint')

    # --- U-Net specific ---
    parser.add_argument('--backbone', default='none',
                         choices=['none', 'resnet18', 'mobilenetv2', 'efficientnetv2'],
                         help='[--model_type unet] Encoder backbone -- must match the '
                              'architecture used to train the checkpoint')
    parser.add_argument('--hidden_channels', default=64, type=int,
                         help='[--model_type unet, --backbone none] U-Net base width')

    parser.add_argument('--agg_to_water', default=True, type=bool,
                         help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')
    parser.add_argument('--batch', default=5, type=int, help='Batch size for evaluation')

    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')

    parser.add_argument('--model_path', default=None, type=str,
                         help='Path to the trained checkpoint (.pth). If omitted, the '
                              'latest epoch checkpoint under trained_models/<variant-or-backbone>/ '
                              'is located automatically, falling back to trained_models/ directly.')
    parser.add_argument('--checkpoint_name', default='best_model.pth', type=str,
                         help='Checkpoint filename inside the epoch folder, used when '
                              'auto-locating --model_path')

    parser.add_argument('--predict_masks', default=True, type=bool,
                         help='Generate test set prediction masks?')
    parser.add_argument('--gen_masks_path',
                         default=os.path.join(PROJECT_ROOT, 'data', 'predicted_masks'),
                         help='Path to where predicted masks are stored')

    parser.add_argument('--tta', default=True, type=bool,
                         help='Average predictions over test-time augmentations '
                              '(rotations x optional horizontal flip), matching the '
                              'augmentation space used during training. Adds extra '
                              'inference cost proportional to the number of passes. '
                              'Set to False for a single plain forward pass, e.g. to '
                              'reproduce previously logged non-TTA results.')
    parser.add_argument('--tta_rotations', default='[0,180]', type=str,
                         help='TTA rotation angles in degrees, as a Python list literal '
                              '(each must be a multiple of 90). Default is a cheap 2-pass '
                              '0/180 sweep; pass "[0,90,180,270]" for the fuller 4- or '
                              '8-pass version (combined with --tta_hflip True).')
    parser.add_argument('--tta_hflip', default=False, type=bool,
                         help='Also average over a horizontal flip at each TTA rotation '
                              '(doubles the pass count). Default is off (rotation-only '
                              'TTA); set True for the fuller rotation+flip sweep.')

    args = parser.parse_args()
    options = vars(args)

    options['tta_rotations'] = ast.literal_eval(options['tta_rotations'])
    if not isinstance(options['tta_rotations'], (list, tuple)):
        options['tta_rotations'] = [options['tta_rotations']]

    main(options)