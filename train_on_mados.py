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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from utils.config import Config
from mssd_net import build_mssd_net
from dataloader import BANDS_MEAN, BANDS_STD, SPECTRAL_INDEX_NAMES, TEXTURE_FEATURE_NAMES, \
    RandomRotationTransform, gen_weights
from mados_dataloader import MADOSDataset, MARIDA_LABELS, MARIDA_CLASSES_NOT_IN_MADOS, remap_native_predictions_to_marida
from precompute_mados_glcm import GLCM_PROPERTIES as GLCM_FEATURE_NAMES
from utils.metrics import Evaluation

# Reuse the exact, already-tested loss classes and LR scheduler builder
# from the MARIDA training script -- training behaviour matches as
# closely as possible; only the data source and class-weight source differ.
from train_swin_unetv2 import DiceLoss, CEDiceLoss, FocalLoss, build_scheduler, ensure_pretrained_checkpoint, build_param_groups

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


def compute_mados_class_distribution(train_dataset, native_15_classes=False, model_output_classes=11):
    """
    One-time pass over the MADOS training split's masks to compute the
    empirical class distribution, for use with gen_weights(). MADOSDataset
    doesn't preload masks into memory the way GenDEBRIS does, so this
    scans the raw mask rasters directly, rather than loading full patches
    through the Dataset __getitem__ (which would also apply augmentation/
    standardization unnecessarily for this one-time count).

    If native_15_classes=True, the distribution is computed over MADOS's
    own 15-class taxonomy directly (no crosswalk, no pixels excluded --
    matches what the model is actually being trained to predict). If
    False (default), matches the original MARIDA-11-class-space behavior.
    """
    from mados_dataloader import remap_mados_mask, MADOS_LABELS

    label_names = MADOS_LABELS if native_15_classes else MARIDA_LABELS
    counts = np.zeros(model_output_classes, dtype=np.int64)
    for roi in tqdm(train_dataset.rois, desc="Scanning MADOS train masks for class distribution"):
        scene_id, crop_id = roi.rsplit('_', 1)
        mask_path = os.path.join(train_dataset.mados_path, scene_id, f'{scene_id}_L2R_cl_{crop_id}.tif')
        ds = train_dataset._gdal.Open(mask_path)
        if ds is None:
            continue
        raw_mask = ds.ReadAsArray().astype(np.int64)
        ds = None
        if native_15_classes:
            remapped = raw_mask - 1  # 1-indexed raw -> 0-indexed native, nothing excluded
        else:
            remapped = remap_mados_mask(raw_mask)
        valid = remapped[remapped != -1]
        if valid.size > 0:
            counts += np.bincount(valid, minlength=model_output_classes)[:model_output_classes]

    total = counts.sum()
    if total == 0:
        raise RuntimeError("No valid (non-ignored, mapped) pixels found anywhere in the MADOS "
                            "training split -- check the crosswalk and file paths before proceeding.")
    distribution = counts.astype(np.float64) / total
    space = "native 15-class" if native_15_classes else "MARIDA 11-class"
    logging.info("MADOS training-split class distribution (%s space): %s", space, dict(zip(label_names, distribution.round(4))))
    print(f"MADOS training-split class distribution ({space} space):")
    for name, freq, count in zip(label_names, distribution, counts):
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
        use_glcm_texture=options['use_glcm_texture'],
        spectral_jitter_prob=options['spectral_jitter_prob'],
        spectral_jitter_strength=options['spectral_jitter_strength'],
        native_15_classes=options['native_15_classes'],
        standardization=standardization, splits_path=options['splits_path'],
    )
    val_dataset = MADOSDataset(
        options['mados_path'], split='val', transform=transform_val,
        use_spectral_indices=options['use_spectral_indices'],
        use_texture_features=options['use_texture_features'],
        use_glcm_texture=options['use_glcm_texture'],
        # native_15_classes deliberately NOT set here -- val/checkpoint-
        # selection always stays in MARIDA's 11-class space, regardless of
        # whether the model is being trained with the auxiliary 15-class
        # signal, so results stay comparable across both modes.
        standardization=standardization, splits_path=options['splits_path'],
    )
    print(f"Loaded {len(train_dataset)} MADOS train patches, {len(val_dataset)} val patches.")
    logging.info("Loaded %d train / %d val MADOS patches.", len(train_dataset), len(val_dataset))

    # ---- Rare-class oversampling ----
    # Plain shuffle=True samples every patch with equal probability. Most
    # patches barely contain Sparse Sargassum/Marine Debris/Foam pixels
    # (Sparse Sargassum in particular has scored under 12% F1 across
    # every architecture/loss combination tested), so the model sees
    # very little of them per epoch even though loss-level class
    # weighting tries to compensate after the fact. A WeightedRandomSampler
    # instead makes patches that DO contain a rare class more likely to
    # be drawn -- a safer alternative to copy-paste augmentation (which
    # was tried and reverted): this only changes how often a real,
    # unmodified patch is drawn, never synthesizes or alters image content.
    if options['oversample_rare']:
        rare_classes = [int(c) for c in options['rare_classes'].split(',') if c.strip() != '']
        sample_weights = train_dataset.compute_sample_weights(rare_classes, boost=options['oversample_boost'])
        logging.info(
            "Rare-class oversampling enabled for classes %s (MARIDA label space, boost=%.1f "
            "per class present). Per-patch weight range: [%.2f, %.2f]",
            rare_classes, options['oversample_boost'], min(sample_weights), max(sample_weights)
        )
        print(f"Rare-class oversampling enabled for classes {rare_classes}, "
              f"weight range [{min(sample_weights):.2f}, {max(sample_weights):.2f}]")
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=options['batch'],
                                   sampler=sampler,  # sampler and shuffle are mutually exclusive
                                   num_workers=options['num_workers'])
    else:
        train_loader = DataLoader(train_dataset, batch_size=options['batch'], shuffle=True,
                                   num_workers=options['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=options['batch'], shuffle=False,
                             num_workers=options['num_workers'])

    # Class weights from MADOS's OWN training distribution -- MARIDA's
    # CLASS_DISTR constant does not apply here, since MADOS's class
    # balance (and complete absence of Clouds) is different.
    #
    # model_output_classes: 15 if training on MADOS's own native taxonomy
    # (native_15_classes=True -- gives the model real learning signal from
    # Oil Spills/Oil Platforms/Jellyfish/Sea Snot pixels too, which
    # otherwise contribute nothing to the loss at all), else 11 (MARIDA's
    # space, the original/default behavior). Checkpoint selection and
    # reported metrics ALWAYS stay in MARIDA's 11-class space either way
    # (see remap_native_predictions_to_marida below) -- this only changes
    # what the model is trained to predict, not what gets reported.
    model_output_classes = 15 if options['native_15_classes'] else NUM_CLASSES
    class_distribution = compute_mados_class_distribution(
        train_dataset, native_15_classes=options['native_15_classes'],
        model_output_classes=model_output_classes
    )
    weight = gen_weights(class_distribution, c=1.02).to(device)

    # IMPORTANT: mirrors MADOSDataset's own internal priority logic exactly
    # -- if both use_texture_features and use_glcm_texture are True, GLCM
    # wins and the fast proxy is skipped (not stacked), so the channel
    # count must reflect that, not naively add both.
    effective_use_texture_features = options['use_texture_features'] and not options['use_glcm_texture']
    n_extra = (len(SPECTRAL_INDEX_NAMES) if options['use_spectral_indices'] else 0) \
        + (len(GLCM_FEATURE_NAMES) if options['use_glcm_texture'] else
           (len(TEXTURE_FEATURE_NAMES) if effective_use_texture_features else 0))
    input_channels = 11 + n_extra

    config = Config()
    swin_cfg = getattr(getattr(config, 'MODEL', None), 'SWIN', None)
    model = build_mssd_net(
        options['variant'],
        img_size=config.DATA.IMG_SIZE,
        patch_size=getattr(swin_cfg, 'PATCH_SIZE', 4),
        in_chans=input_channels,
        num_classes=model_output_classes,
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
        criterion = DiceLoss(num_classes=model_output_classes, weight=weight, ignore_index=-1)
    elif options['loss_type'] == 'ce_dice':
        criterion = CEDiceLoss(num_classes=model_output_classes, weight=weight, ignore_index=-1,
                                dice_weight=options['dice_weight'])
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=weight)
    logging.info("Using loss_type=%s", options['loss_type'])

    if options['encoder_lr_mult'] != 1.0:
        param_groups, n_encoder, n_other = build_param_groups(
            model, options['lr'], options['encoder_lr_mult'], options['decay']
        )
        optimizer = torch.optim.Adam(param_groups)
        logging.info(
            "Differential LR enabled: %d pretrained-encoder params at lr=%.2e, "
            "%d other params (decoder/MSSD/random-init) at lr=%.2e",
            n_encoder, options['lr'] * options['encoder_lr_mult'], n_other, options['lr']
        )
    else:
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
                    # Reshape using the model's ACTUAL output channel count
                    # (15 in native_15_classes mode, 11 otherwise) -- using
                    # the fixed NUM_CLASSES=11 here would crash in native
                    # mode, since the model genuinely outputs 15 channels.
                    probs = torch.movedim(probs, 1, -1).reshape(-1, model_output_classes)
                    targets = targets.reshape(-1)
                    mask = targets != -1
                    probs, targets = probs[mask], targets[mask]

                    pred_classes = probs.cpu().numpy().argmax(1)
                    if options['native_15_classes']:
                        # val_dataset's targets are already in MARIDA's
                        # 11-class space (native_15_classes=False for
                        # val_dataset, by design -- see its construction
                        # above), but the model's predictions are still in
                        # native 15-class space, so remap predictions down
                        # before comparing. Predictions of a class with no
                        # MARIDA equivalent (Oil Spills etc.) become -1 and
                        # must be excluded here too, same as targets==-1.
                        pred_classes = remap_native_predictions_to_marida(pred_classes)
                        valid_pred = pred_classes != -1
                        pred_classes = pred_classes[valid_pred]
                        targets_np = targets.cpu().numpy()[valid_pred]
                    else:
                        targets_np = targets.cpu().numpy()

                    y_pred += pred_classes.tolist()
                    y_true += targets_np.tolist()

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
                save_path = os.path.join(checkpoint_dir, 'best_model.pth')
                backup_path = os.path.join(checkpoint_dir, 'best_model_prev.pth')
                # Keep one step-back safety copy (in case a save gets
                # interrupted mid-write) without letting checkpoints
                # accumulate across every improving epoch -- the old
                # per-epoch-folder scheme filled Kaggle's disk quota by
                # epoch 200 on a long run and crashed torch.save() mid-write.
                if os.path.exists(save_path):
                    try:
                        os.replace(save_path, backup_path)
                    except OSError:
                        pass  # non-fatal; proceed to overwrite save_path anyway
                torch.save(model.state_dict(), save_path)
                logging.info("Best model improved (macroF1 excl. Clouds = %.4f, mIoU excl. Clouds = %.4f) "
                              "at epoch %d. Saved to: %s",
                              current_score, current_iou, epoch, save_path)
                print(f"Best model (epoch {epoch}) saved to: {save_path}")
            else:
                early_stop_counter += 1
                logging.info("No improvement for %d/%d epochs (best so far: macroF1=%.4f at an earlier epoch).",
                              early_stop_counter, options['patience'], best_score)
                print(f"No improvement for {early_stop_counter}/{options['patience']} epochs "
                      f"(best so far: macroF1={best_score:.4f}).")
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
    parser.add_argument('--use_texture_features', default=False, type=bool,
                         help="Fast local std-dev + gradient magnitude texture proxy, computed "
                              "live. Ignored if --use_glcm_texture is also True (GLCM wins).")
    parser.add_argument('--native_15_classes', default=False, type=bool,
                         help="Train on MADOS's own 15-class taxonomy directly, instead of "
                              "remapping every mask down to MARIDA's 11 classes. Gives the "
                              "model real learning signal from the ~4 extra classes' pixels "
                              "(Oil Spills, Oil Platforms, Jellyfish, Sea Snot) that are "
                              "otherwise thrown away as ignore_index -- a form of auxiliary-"
                              "task learning that may improve shared feature representations "
                              "even for the 11 classes actually reported. Model output layer "
                              "becomes 15 channels. Checkpoint selection and reported metrics "
                              "STILL happen in MARIDA's 11-class space (predictions are "
                              "remapped down for scoring, via remap_native_predictions_to_marida) "
                              "so results stay directly comparable to non-native runs. "
                              "Default False = original behavior, unchanged.")
    parser.add_argument('--use_glcm_texture', default=False, type=bool,
                         help="Use TRUE precomputed GLCM texture features (Contrast, "
                              "Dissimilarity, Homogeneity, Energy, Correlation, ASM) instead of "
                              "the fast proxy. REQUIRES running precompute_mados_glcm.py "
                              "--mados_path <same path> FIRST -- this flag only loads cached "
                              "<scene>_L2R_glcm_<crop>.tif files, it does not compute them. "
                              "Matches MARIDA's own RF feature set (their single most important "
                              "feature, per their own feature-importance analysis).")
    parser.add_argument('--spectral_jitter_prob', default=0.0, type=float,
                         help="Probability, per training sample, of multiplying each raw band "
                              "by an independent random factor (see --spectral_jitter_strength), "
                              "simulating realistic Sentinel-2 sensor/atmospheric variation. "
                              "Same technique as GenDEBRIS's MARIDA pipeline. Train-only. "
                              "Default 0.0 = off.")
    parser.add_argument('--spectral_jitter_strength', default=0.05, type=float,
                         help="Each band's multiplicative jitter factor is drawn uniformly from "
                              "[1-strength, 1+strength]. Only used if --spectral_jitter_prob > 0.")
    parser.add_argument('--oversample_rare', default=False, type=bool,
                         help="Use a WeightedRandomSampler that oversamples training patches "
                              "containing --rare_classes, instead of plain shuffle=True. "
                              "Matches train_swin_unetv2.py's MARIDA convention. Default True.")
    parser.add_argument('--rare_classes', default='0,2,3,8', type=str,
                         help="Comma-separated MARIDA-label-space class indices (e.g. "
                              "'0,2,3,8' = Marine Debris, Sparse Sargassum, Natural Organic "
                              "Material, Foam) to oversample. Same convention as "
                              "train_swin_unetv2.py's --rare_classes. Only used if "
                              "--oversample_rare is True.")
    parser.add_argument('--oversample_boost', default=5.0, type=float,
                         help="Extra sampling weight added per rare class present in a patch "
                              "(see compute_sample_weights). Only used if --oversample_rare.")
    parser.add_argument('--use_pretrained', default=True, type=bool,
                         help='Load ImageNet-pretrained Swin V2 weights (same channel-adaptation '
                              'logic as the MARIDA training script -- works regardless of '
                              'in_chans, verified earlier in this project).')
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--encoder_lr_mult', default=1.0, type=float,
                         help="Multiplier applied to --lr for the ImageNet-pretrained encoder "
                              "params (patch_embed + layers.*); everything else (decoder, MSSD "
                              "modules, and the randomly-initialized relative-position-bias "
                              "tensors) uses --lr directly. Default 1.0 = single shared LR "
                              "(previous behavior, unchanged unless explicitly set). Try e.g. "
                              "0.1 so pretrained ImageNet features move slower than the "
                              "from-scratch parts. Reuses build_param_groups from "
                              "train_swin_unetv2.py, verified against the real MSSDNet model.")
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