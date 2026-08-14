# -*- coding: utf-8 -*-
"""
train_on_mados.py

Train a NEW MSSD-Net model FROM SCRATCH using only MADOS data (not
merged with MARIDA), reporting in MARIDA's 11-class label space (via
mados_dataloader.py's crosswalk) so this can be directly compared to
your MARIDA-trained models using the exact same evaluation script
(evaluate_on_mados.py --dataset marida or --dataset mados).

This reuses your already-tested loss classes (DiceLoss, CEDiceLoss,
FocalLoss) and LR scheduler (build_scheduler) directly from
train_swin_unetv2.py, so training behaviour matches your MARIDA
pipeline as closely as possible -- the only things that differ are the
data source and (necessarily) the class-weight computation, which is
now derived from MADOS's own training-split class distribution instead
of MARIDA's.

======================================================================
IMPORTANT LIMITATIONS -- read before running
======================================================================
  - Clouds (MARIDA class 5) NEVER appears in MADOS at all. A model
    trained here will never see a single Clouds example, and cannot
    learn to predict it meaningfully. Checkpoint selection (validation
    Macro F1) EXCLUDES Clouds accordingly, matching evaluate_on_mados.py.
    If you then evaluate this checkpoint on MARIDA's own test set
    (--dataset marida), expect Clouds to score ~0% -- this is not a
    bug, it is exactly what "trained only on MADOS" predicts.
  - 4 MADOS classes (Oil Spills, Oil Platforms, Jellyfish, Sea Snot)
    have no MARIDA equivalent and are excluded entirely (ignore_index),
    so this model also never learns anything about them, even though
    they exist in the raw MADOS data.
  - File-path/naming assumptions for MADOS are UNVERIFIED against a
    real download -- see mados_dataloader.py's module docstring.

USAGE:
    python train_on_mados.py --mados_path /path/to/MADOS \
        --loss_type ce_dice --use_spectral_indices True --use_texture_features True \
        --epochs 300 --patience 35
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from utils.config import Config
from mssd_net import build_mssd_net
from dataloader import BANDS_MEAN, BANDS_STD, SPECTRAL_INDEX_NAMES, TEXTURE_FEATURE_NAMES, \
    RandomRotationTransform, gen_weights
from mados_dataloader import MADOSDataset, MARIDA_LABELS, MARIDA_CLASSES_NOT_IN_MADOS
from utils.metrics import Evaluation

# Reuse the exact, already-tested loss classes and LR scheduler builder
# from the MARIDA training script -- training behaviour matches as
# closely as possible; only the data source and class-weight source differ.
from train_swin_unetv2 import DiceLoss, CEDiceLoss, FocalLoss, build_scheduler, ensure_pretrained_checkpoint

os.makedirs(os.path.join(PROJECT_ROOT, 'trained_models'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

NUM_CLASSES = 11  # MARIDA's label space, via the crosswalk
CLOUDS_INDEX = MARIDA_LABELS.index('Clouds')


def seed_all(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mados_class_distribution(train_dataset):
    """
    One-time pass over the MADOS training split's masks to compute the
    empirical class distribution in MARIDA's 11-class space (after the
    crosswalk remapping), for use with gen_weights(). MADOSDataset
    doesn't preload masks into memory the way GenDEBRIS does, so this
    scans the raw mask rasters directly and reuses the same remapping
    logic, rather than loading full patches through the Dataset
    __getitem__ (which would also apply augmentation/standardization
    unnecessarily for this one-time count).
    """
    from mados_dataloader import remap_mados_mask

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for roi in tqdm(train_dataset.rois, desc="Scanning MADOS train masks for class distribution"):
        scene_id, crop_id = roi.rsplit('_', 1)
        mask_path = os.path.join(train_dataset.mados_path, scene_id, f'{scene_id}_L2R_cl_{crop_id}.tif')
        ds = train_dataset._gdal.Open(mask_path)
        if ds is None:
            continue
        raw_mask = ds.ReadAsArray().astype(np.int64)
        ds = None
        remapped = remap_mados_mask(raw_mask)
        valid = remapped[remapped != -1]
        if valid.size > 0:
            counts += np.bincount(valid, minlength=NUM_CLASSES)[:NUM_CLASSES]

    total = counts.sum()
    if total == 0:
        raise RuntimeError("No valid (non-ignored, mapped) pixels found anywhere in the MADOS "
                            "training split -- check the crosswalk and file paths before proceeding.")
    distribution = counts.astype(np.float64) / total
    logging.info("MADOS training-split class distribution (MARIDA label space): %s", dict(zip(MARIDA_LABELS, distribution.round(4))))
    print("MADOS training-split class distribution (MARIDA label space):")
    for name, freq, count in zip(MARIDA_LABELS, distribution, counts):
        print(f"  {name:28s}: {freq*100:6.2f}%  ({count} pixels)")
    return torch.tensor(distribution, dtype=torch.float32)


def macro_f1_excluding_clouds(metrics_dict, y_true=None, y_pred=None):
    """
    Checkpoint-selection metric: Macro F1 computed EXCLUDING Clouds,
    since MADOS never contains a single Clouds pixel -- including it
    would just average in a permanent, meaningless 0 (or worse, an
    undefined value) that has nothing to do with how well training is
    actually going on classes MADOS can teach the model about.
    """
    # utils.metrics.Evaluation() already returns an overall macroF1 across
    # all classes present; for an exact per-class exclusion we recompute
    # from sklearn directly here using the same y_true/y_pred sklearn
    # would use, restricted to the testable label set.
    from sklearn.metrics import f1_score
    testable_labels = [i for i in range(NUM_CLASSES) if i != CLOUDS_INDEX]
    return f1_score(y_true, y_pred, labels=testable_labels, average='macro', zero_division=0)


def macro_iou_excluding_clouds(y_true, y_pred):
    """
    Same idea as macro_f1_excluding_clouds, but for mIoU -- reported
    alongside Macro F1 at every validation step, since IoU is the
    project's other headline metric throughout (matching
    evaluate_on_mados.py's macro_iou_excluding_clouds output) and was
    previously missing from this script's per-epoch training log.
    """
    from sklearn.metrics import jaccard_score
    testable_labels = [i for i in range(NUM_CLASSES) if i != CLOUDS_INDEX]
    return jaccard_score(y_true, y_pred, labels=testable_labels, average='macro', zero_division=0)


def main(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(0)

    log_path = os.path.join(PROJECT_ROOT, 'logs', 'training_on_mados.log')
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    logging.info('*' * 10)
    logging.info('Training MSSD-Net from scratch on MADOS only. loss_type=%s, '
                  'use_spectral_indices=%s, use_texture_features=%s',
                  options['loss_type'], options['use_spectral_indices'], options['use_texture_features'])
    logging.info('NOTE: Clouds excluded from checkpoint-selection Macro F1 -- MADOS has no Clouds data.')

    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)
    transform_steps = [transforms.ToTensor()]
    if options['train_rotations']:
        transform_steps.append(RandomRotationTransform([-90, 0, 90, 180]))
    if options['train_hflip']:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_train = transforms.Compose(transform_steps)
    transform_val = transforms.Compose([transforms.ToTensor()])

    train_dataset = MADOSDataset(
        options['mados_path'], split='train', transform=transform_train,
        use_spectral_indices=options['use_spectral_indices'],
        use_texture_features=options['use_texture_features'],
        standardization=standardization, splits_path=options['splits_path'],
    )
    val_dataset = MADOSDataset(
        options['mados_path'], split='val', transform=transform_val,
        use_spectral_indices=options['use_spectral_indices'],
        use_texture_features=options['use_texture_features'],
        standardization=standardization, splits_path=options['splits_path'],
    )
    print(f"Loaded {len(train_dataset)} MADOS train patches, {len(val_dataset)} val patches.")
    logging.info("Loaded %d train / %d val MADOS patches.", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=options['batch'], shuffle=True,
                               num_workers=options['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=options['batch'], shuffle=False,
                             num_workers=options['num_workers'])

    # Class weights from MADOS's OWN training distribution -- MARIDA's
    # CLASS_DISTR constant does not apply here, since MADOS's class
    # balance (and complete absence of Clouds) is different.
    class_distribution = compute_mados_class_distribution(train_dataset)
    weight = gen_weights(class_distribution, c=1.02).to(device)

    n_extra = (len(SPECTRAL_INDEX_NAMES) if options['use_spectral_indices'] else 0) \
        + (len(TEXTURE_FEATURE_NAMES) if options['use_texture_features'] else 0)
    input_channels = 11 + n_extra

    config = Config()
    swin_cfg = getattr(getattr(config, 'MODEL', None), 'SWIN', None)
    model = build_mssd_net(
        options['variant'],
        img_size=config.DATA.IMG_SIZE,
        patch_size=getattr(swin_cfg, 'PATCH_SIZE', 4),
        in_chans=input_channels,
        num_classes=NUM_CLASSES,
        embed_dim=getattr(swin_cfg, 'EMBED_DIM', 96),
        depths=getattr(swin_cfg, 'DEPTHS', [2, 2, 2, 2]),
        depths_decoder=getattr(swin_cfg, 'DEPTHS_DECODER', [1, 2, 2, 2]),
        num_heads=getattr(swin_cfg, 'NUM_HEADS', [3, 6, 12, 24]),
        window_size=getattr(swin_cfg, 'WINDOW_SIZE', 8),
    )
    if options['use_pretrained']:
        pretrained_path = ensure_pretrained_checkpoint(options.get('pretrained_path'))
        model.load_pretrained(pretrained_path)
    model.to(device)

    if options['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=weight, gamma=options['focal_gamma'], ignore_index=-1, reduction='mean')
    elif options['loss_type'] == 'dice':
        criterion = DiceLoss(num_classes=NUM_CLASSES, weight=weight, ignore_index=-1)
    elif options['loss_type'] == 'ce_dice':
        criterion = CEDiceLoss(num_classes=NUM_CLASSES, weight=weight, ignore_index=-1,
                                dice_weight=options['dice_weight'])
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=weight)
    logging.info("Using loss_type=%s", options['loss_type'])

    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['decay'])
    warmup_scheduler, main_scheduler, warmup_steps = build_scheduler(
        optimizer, options, steps_per_epoch=len(train_loader)
    )

    best_score = float('-inf')
    early_stop_counter = 0
    global_step = 0

    checkpoint_dir = os.path.join(PROJECT_ROOT, 'trained_models', 'mados_only', options['variant'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, options['epochs'] + 1):
        model.train()
        train_losses = []
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch} training"):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options['grad_clip'])
            optimizer.step()

            if warmup_scheduler is not None and global_step < warmup_steps:
                warmup_scheduler.step()
            global_step += 1

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / max(1, len(train_losses))

        if epoch % options['eval_every'] == 0 or epoch == 1:
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    logits = model(images)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    probs = torch.movedim(probs, 1, -1).reshape(-1, NUM_CLASSES)
                    targets = targets.reshape(-1)
                    mask = targets != -1
                    probs, targets = probs[mask], targets[mask]
                    y_pred += probs.cpu().numpy().argmax(1).tolist()
                    y_true += targets.cpu().numpy().tolist()

            if not y_true:
                logging.warning("Epoch %d: no valid val pixels found -- skipping checkpoint check.", epoch)
                continue

            current_score = macro_f1_excluding_clouds(None, y_true, y_pred)
            current_iou = macro_iou_excluding_clouds(y_true, y_pred)
            logging.info("Epoch %d - Train loss: %.4f - Val macroF1 (excl. Clouds): %.4f - Val mIoU (excl. Clouds): %.4f",
                          epoch, avg_train_loss, current_score, current_iou)
            print(f"Epoch {epoch} - Train loss: {avg_train_loss:.4f} - "
                  f"Val macroF1 (excl. Clouds): {current_score:.4f} - "
                  f"Val mIoU (excl. Clouds): {current_iou:.4f}")

            if current_score > best_score:
                best_score = current_score
                early_stop_counter = 0
                save_dir = os.path.join(checkpoint_dir, str(epoch))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                logging.info("Best model improved (macroF1 excl. Clouds = %.4f, mIoU excl. Clouds = %.4f). Saved to: %s",
                              current_score, current_iou, save_path)
                print(f"Best model saved to: {save_path}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= options['patience']:
                    logging.info("Early stopping triggered after epoch %d.", epoch)
                    print("Early stopping triggered.")
                    break

            # ---- Step main scheduler once per epoch, after warmup ----
            # (matches train_swin_unetv2.py's tested pattern exactly)
            if global_step >= warmup_steps:
                if options['scheduler'] == 'plateau':
                    # ReduceLROnPlateau's default mode='min' expects a
                    # loss-like quantity that should decrease -- macroF1
                    # should INCREASE, so pass its negative.
                    main_scheduler.step(-current_score)
                else:
                    main_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mados_path', required=True,
                         help="Path to the STACKED MADOS data (i.e. the '<something>_nearest' "
                              "folder produced by MADOS's own utils/stack_patches.py -- e.g. "
                              "MADOS_nearest, NOT the original raw MADOS/ folder).")
    parser.add_argument('--splits_path', default=None,
                         help="Path to the folder containing {train,val,test}_X.txt. Defaults "
                              "to '<mados_path>/splits' if not given, but splits/ is normally "
                              "only created in the ORIGINAL (unstacked) MADOS/ folder, not in "
                              "the stacked MADOS_nearest/ folder -- so this usually needs to be "
                              "set explicitly, e.g. --splits_path /path/to/MADOS/splits.")
    parser.add_argument('--variant', default='baseline',
                         choices=['baseline', '+dilated', '+dilated+attention', 'full',
                                  'only_dilated', 'only_attention', 'only_residual'])
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal', 'dice', 'ce_dice'])
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--dice_weight', default=0.5, type=float)
    parser.add_argument('--use_spectral_indices', default=False, type=bool)
    parser.add_argument('--use_texture_features', default=False, type=bool)
    parser.add_argument('--use_pretrained', default=True, type=bool,
                         help='Load ImageNet-pretrained Swin V2 weights (same channel-adaptation '
                              'logic as the MARIDA training script -- works regardless of '
                              'in_chans, verified earlier in this project).')
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--scheduler', default='sgdr', choices=['sgdr', 'plateau', 'multistep', 'cosine'])
    parser.add_argument('--lr_steps', default=[45, 65], type=list)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--train_rotations', default=True, type=bool)
    parser.add_argument('--train_hflip', default=True, type=bool)
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    args = parser.parse_args()
    options = vars(args)
    main(options)