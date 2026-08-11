# -*- coding: utf-8 -*-
"""
Training script for Swin-UNet V2 (MSSD-Net) on the MARIDA dataset.

This script has been aligned with train_unet.py so that both models are
trained under an IDENTICAL protocol (epochs, batch size, optimizer, weight
decay, warmup, LR schedule, gradient clipping, early stopping, loss
weighting, and data-loading settings). Only architecture-specific options
(--pretrained_path here; --hidden_channels for the U-Net) differ between
the two scripts.

Changes from the previous version of this script:
  * epochs default 300 -> 100 (matches the U-Net script and the paper)
  * batch default 8 -> 16
  * lr default 5e-5 -> 1e-4
  * patience default 15 -> 10
  * scheduler default 'plateau' -> 'sgdr', and 'sgdr' (true
    CosineAnnealingWarmRestarts) added as a real option -- the previous
    'cosine' choice was plain CosineAnnealingLR, which is NOT SGDR despite
    the paper describing "stochastic gradient descent with warm restarts".
  * fixed a scheduler-stepping bug: when warmup was combined with the
    'cosine'/'multistep' choices via SequentialLR, the main scheduler was
    never advanced after the warmup phase ended. The warmup and main
    schedulers are now stepped explicitly and separately.
"""

import argparse
import ast
import json
import logging
import os
import random
import sys
from os.path import dirname as up

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import (
    LinearLR, ReduceLROnPlateau, CosineAnnealingLR,
    CosineAnnealingWarmRestarts, MultiStepLR,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = up(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.config import Config
from mssd_net import build_mssd_net, VARIANTS
from dataloader import (
    GenDEBRIS, BANDS_MEAN, BANDS_STD, DATASET_PATH,
    RandomRotationTransform, gen_weights, CLASS_DISTR
)

sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))
from utils.metrics import Evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logging.info('*' * 10)

# ----------------------------------------------------------------------
# ImageNet-pretrained Swin V2 (Swin-UNet V2 encoder) checkpoint.
# Auto-downloaded on first use so --pretrained_path does not have to be
# fetched by hand. Source: official microsoft/Swin-Transformer release.
# ----------------------------------------------------------------------
SWINV2_TINY_URL = ('https://github.com/SwinTransformer/storage/releases/'
                    'download/v2.0.0/swinv2_tiny_patch4_window8_256.pth')
SWINV2_TINY_FILENAME = 'swinv2_tiny_patch4_window8_256.pth'


def ensure_pretrained_checkpoint(path=None, url=SWINV2_TINY_URL):
    """Return a local path to the ImageNet Swin V2 checkpoint, downloading
    it from the official release if it is not already on disk.

    If `path` is given and already exists, it is used as-is (no download).
    If `path` is given but missing, the file is downloaded to that exact
    path (creating parent directories as needed). If `path` is None, the
    checkpoint is downloaded to a default cache location under
    <PROJECT_ROOT>/pretrained/.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, 'pretrained', SWINV2_TINY_FILENAME)

    if os.path.isfile(path):
        logging.info('Found existing pretrained checkpoint at: %s', path)
        return path

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    logging.info('Pretrained checkpoint not found locally. Downloading from %s', url)
    logging.info('Saving to: %s', path)
    torch.hub.download_url_to_file(url, path, progress=True)
    logging.info('Download complete.')
    return path


def seed_all(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Worker initialisation for DataLoader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ConfidenceWeightedCE(nn.Module):
    """
    Cross-entropy loss with an optional additional per-pixel weight (e.g.
    annotation-confidence weight from GenDEBRIS, see --use_confidence_weighting),
    applied on top of the existing per-class weight vector. Drop-in
    replacement for nn.CrossEntropyLoss(weight=..., ignore_index=...) when
    pixel_weight=None (the default): reduces to the exact same masked mean.
    """

    def __init__(self, weight=None, ignore_index=-1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
        self.ignore_index = ignore_index

    def forward(self, logits, targets, pixel_weight=None):
        per_pixel = self.ce(logits, targets)  # (B, H, W); 0 at ignored positions
        valid = (targets != self.ignore_index).float()
        w = valid if pixel_weight is None else valid * pixel_weight
        denom = w.sum().clamp_min(1e-8)
        return (per_pixel * w).sum() / denom


class FocalLoss(nn.Module):
    """Focal Loss for multi-class segmentation."""

    def __init__(self, alpha=None, gamma=2, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets, pixel_weight=None):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none',
            weight=self.alpha, ignore_index=self.ignore_index
        )
        valid_mask = (targets != self.ignore_index).float()
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        w = valid_mask if pixel_weight is None else valid_mask * pixel_weight
        focal_loss = focal_loss * w

        if self.reduction == 'mean':
            denom = w.sum()
            if denom > 0:
                return focal_loss.sum() / denom
            return torch.tensor(0.0, device=inputs.device)
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation, optionally class-weighted
    with the same weight vector used for CrossEntropyLoss (--weight_param),
    and optionally further weighted per-pixel by annotation confidence
    (see --use_confidence_weighting).

    Unlike raw IoU/Jaccard loss, Dice's gradient is gentler on
    near-empty-overlap (rare-class) pixels, which tends to train more
    stably than Focal loss under MARIDA's severe imbalance -- Focal
    (gamma=2) measurably hurt Macro F1 in our ablation for both the
    baseline and residual-fusion variants.
    """

    def __init__(self, num_classes, weight=None, ignore_index=-1, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets, pixel_weight=None):
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        valid = (targets != self.ignore_index)
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0  # placeholder class, masked out below
        one_hot = F.one_hot(targets_clamped, self.num_classes).permute(0, 3, 1, 2).float()

        valid_f = valid.unsqueeze(1).float()
        # A single weight factor applied consistently to the intersection
        # AND both union terms below (not pre-multiplied into probs/one_hot
        # separately, which would double-count it in the intersection term).
        w = valid_f if pixel_weight is None else valid_f * pixel_weight.unsqueeze(1)

        dims = (0, 2, 3)
        intersection = (w * probs * one_hot).sum(dims)
        union = (w * probs).sum(dims) + (w * one_hot).sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)

        if self.weight is not None:
            return 1 - (dice_per_class * self.weight).sum() / self.weight.sum()
        return 1 - dice_per_class.mean()


class CEDiceLoss(nn.Module):
    """Weighted sum of CrossEntropyLoss and DiceLoss: CE + dice_weight * Dice."""

    def __init__(self, num_classes, weight=None, ignore_index=-1, dice_weight=0.5):
        super().__init__()
        self.ce = ConfidenceWeightedCE(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, weight=weight, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, targets, pixel_weight=None):
        return self.ce(logits, targets, pixel_weight=pixel_weight) \
            + self.dice_weight * self.dice(logits, targets, pixel_weight=pixel_weight)


def build_param_groups(model, base_lr, encoder_lr_mult, weight_decay):
    """
    Split model parameters into two LR groups: the ImageNet-pretrained
    encoder (patch_embed + layers -- see MSSDNet.load_pretrained) gets
    base_lr * encoder_lr_mult, everything else (decoder, MSSD modules,
    and the randomly-initialized relative-position-bias tensors that
    live inside `layers` but have no ImageNet counterpart) gets base_lr.

    This targets a real, logged asymmetry: MSSDNet.load_pretrained()
    reports ~99 tensors loaded from ImageNet but ~188 left at random
    init, so part of the network starts pretrained and part starts from
    scratch. Sharing one LR across both is a poor fit for that -- a
    lower LR protects the pretrained features from being overwritten
    too fast early in training, while the from-scratch parts can move
    at the full rate.

    Returns:
        (list[dict], int, int): param groups for torch.optim.Adam, plus
        the encoder/other parameter counts (for logging).
    """
    encoder_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('patch_embed.') or name.startswith('layers.'):
            encoder_params.append(param)
        else:
            other_params.append(param)

    groups = [
        {'params': encoder_params, 'lr': base_lr * encoder_lr_mult, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay},
    ]
    return groups, len(encoder_params), len(other_params)


def build_scheduler(optimizer, options, steps_per_epoch):
    """
    Build a linear-warmup scheduler plus a main scheduler. Identical
    construction to train_unet.py so both scripts follow the same LR
    schedule for a given --scheduler choice.

    Returns (warmup_scheduler_or_None, main_scheduler, warmup_steps).
    """
    warmup_steps = options['warmup_epochs'] * steps_per_epoch

    warmup_scheduler = None
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )

    if options['scheduler'] == 'sgdr':
        # True SGDR: cosine annealing with warm restarts, doubling the
        # restart period each cycle (T_0=10 epochs, T_mult=2 -> restarts
        # at epochs 10, 30, 70, ... after warmup).
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    elif options['scheduler'] == 'cosine':
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, options['epochs'] - options['warmup_epochs']),
            eta_min=1e-6,
        )
    elif options['scheduler'] == 'multistep':
        main_scheduler = MultiStepLR(optimizer, options['lr_steps'], gamma=0.1)
    else:  # 'plateau'
        main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    return warmup_scheduler, main_scheduler, warmup_steps


def main(options):
    """Main training loop."""
    seed_all(0)
    generator = torch.Generator()
    generator.manual_seed(0)

    best_score = float('-inf')
    early_stop_counter = 0
    patience = options['patience']
    writer = SummaryWriter(os.path.join(PROJECT_ROOT, 'logs', options['tensorboard']))

    transform_steps = [transforms.ToTensor()]
    if options['train_rotations']:
        transform_steps.append(RandomRotationTransform(options['train_rotations']))
    if options['train_hflip']:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_train = transforms.Compose(transform_steps)
    transform_test = transforms.Compose([transforms.ToTensor()])

    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)

    if options['mode'] == 'train':
        splits_dir = os.path.join(DATASET_PATH, 'splits')
        train_rois_raw = np.genfromtxt(os.path.join(splits_dir, 'train_X.txt'), dtype='str')
        val_rois_raw = np.genfromtxt(os.path.join(splits_dir, 'val_X.txt'), dtype='str')

        logging.info(
            "Patches before augmentation - train: %d, val: %d",
            len(train_rois_raw), len(val_rois_raw)
        )
        print(f"Patches before augmentation - train: {len(train_rois_raw)}, val: {len(val_rois_raw)}")

        # Parsed once, reused by both copy-paste augmentation (GenDEBRIS)
        # and rare-class oversampling (WeightedRandomSampler) below, so the
        # two augmentations always target the same class list.
        rare_classes = [int(c) for c in options['rare_classes'].split(',') if c.strip() != '']

        train_dataset = GenDEBRIS(
            'train', transform=transform_train, standardization=standardization,
            agg_to_water=options['agg_to_water'],
            rare_classes=rare_classes,
            copy_paste_prob=options['copy_paste_prob'],
            spectral_jitter_prob=options['spectral_jitter_prob'],
            spectral_jitter_strength=options['spectral_jitter_strength'],
            use_spectral_indices=options['use_spectral_indices'],
            use_texture_features=options['use_texture_features'],
            use_confidence_weighting=options['use_confidence_weighting'],
        )
        val_dataset = GenDEBRIS(
            'val', transform=transform_test, standardization=standardization,
            agg_to_water=options['agg_to_water'],
            use_spectral_indices=options['use_spectral_indices'],
            use_texture_features=options['use_texture_features'],
            use_confidence_weighting=options['use_confidence_weighting'],
            # No rare_classes / copy_paste_prob / spectral_jitter_prob here:
            # those are train-only augmentations. use_spectral_indices,
            # use_texture_features, and use_confidence_weighting are all
            # feature/loss representations, not augmentations, so they
            # MUST match train -- val/test need the same input channels
            # and (if enabled) confidence weights as what the model was
            # built and trained with.
        )

        logging.info(
            "Patches after augmentation - train: %d, val: %d",
            len(train_dataset), len(val_dataset)
        )
        print(f"Patches after augmentation - train: {len(train_dataset)}, val: {len(val_dataset)}")
        num_input_channels = train_dataset.num_channels

        # ---- Rare-class oversampling ----
        # Plain shuffle=True samples every patch with equal probability.
        # Most patches barely contain Marine Debris/Sparse Sargassum/Foam
        # pixels, so the model sees very little of them per epoch even
        # though the loss weighting tries to compensate after the fact.
        # A WeightedRandomSampler instead makes patches that DO contain a
        # rare class more likely to be drawn, changing what the model
        # actually sees during training rather than just how much a
        # mistake on it costs.
        if options['oversample_rare']:
            sample_weights = train_dataset.compute_sample_weights(
                rare_classes, boost=options['oversample_boost']
            )
            logging.info(
                "Rare-class oversampling enabled for classes %s (boost=%.1f per class present). "
                "Per-patch weight range: [%.2f, %.2f]",
                rare_classes, options['oversample_boost'],
                min(sample_weights), max(sample_weights)
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True,
                generator=generator
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=options['batch'],
                sampler=sampler,  # sampler and shuffle are mutually exclusive in DataLoader
                num_workers=options['num_workers'],
                pin_memory=options['pin_memory'],
                prefetch_factor=options['prefetch_factor'],
                persistent_workers=options['persistent_workers'],
                worker_init_fn=seed_worker,
                generator=generator
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=options['batch'],
                shuffle=True,
                num_workers=options['num_workers'],
                pin_memory=options['pin_memory'],
                prefetch_factor=options['prefetch_factor'],
                persistent_workers=options['persistent_workers'],
                worker_init_fn=seed_worker,
                generator=generator
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=options['batch'],
            shuffle=False,
            num_workers=options['num_workers'],
            pin_memory=options['pin_memory'],
            prefetch_factor=options['prefetch_factor'],
            persistent_workers=options['persistent_workers'],
            worker_init_fn=seed_worker,
            generator=generator
        )

    elif options['mode'] == 'test':
        test_dataset = GenDEBRIS(
            'test', transform=transform_test, standardization=standardization,
            agg_to_water=options['agg_to_water'],
            use_spectral_indices=options['use_spectral_indices'],
            use_texture_features=options['use_texture_features'],
            use_confidence_weighting=options['use_confidence_weighting'],
        )
        num_input_channels = test_dataset.num_channels
        test_loader = DataLoader(
            test_dataset,
            batch_size=options['batch'],
            shuffle=False,
            num_workers=options['num_workers'],
            pin_memory=options['pin_memory'],
            prefetch_factor=options['prefetch_factor'],
            persistent_workers=options['persistent_workers'],
            worker_init_fn=seed_worker,
            generator=generator
        )
    else:
        raise ValueError("mode must be 'train' or 'test'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Model. Reads Swin architecture settings from Config() where
    # available, falling back to the values we validated end-to-end
    # (window_size=8 is required for img_size=256 with patch_size=4 --
    # window_size=7, the SwinTransformerSys default, does NOT divide the
    # 8x8 bottleneck resolution and will crash the forward pass).
    # ------------------------------------------------------------------
    config = Config()
    swin_cfg = getattr(config, 'MODEL', None)
    swin_cfg = getattr(swin_cfg, 'SWIN', None)

    model = build_mssd_net(
        options['variant'],
        img_size=config.DATA.IMG_SIZE,
        patch_size=getattr(swin_cfg, 'PATCH_SIZE', 4),
        in_chans=num_input_channels,  # from GenDEBRIS.num_channels: 11, or 17 with --use_spectral_indices
        num_classes=options['output_channels'],
        embed_dim=getattr(swin_cfg, 'EMBED_DIM', 96),
        depths=getattr(swin_cfg, 'DEPTHS', [2, 2, 2, 2]),
        depths_decoder=getattr(swin_cfg, 'DEPTHS_DECODER', [1, 2, 2, 2]),
        num_heads=getattr(swin_cfg, 'NUM_HEADS', [3, 6, 12, 24]),
        window_size=getattr(swin_cfg, 'WINDOW_SIZE', 8),
    )
    logging.info("Model variant: %s (%s)", options['variant'], VARIANTS[options['variant']])
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Total parameters: %s", f"{n_params:,}")
    model.to(device)

    if options['resume_from_epoch'] > 1:
        resume_dir = os.path.join(options['checkpoint_path'], str(options['resume_from_epoch']))
        model_file = os.path.join(resume_dir, options['checkpoint_name'])
        logging.info('Resuming training from epoch %d', options['resume_from_epoch'])
        logging.info('Loading model from: %s', model_file)
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info('Model loaded successfully.')
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif options.get('use_pretrained', True):
        pretrained_path = ensure_pretrained_checkpoint(options.get('pretrained_path'))
        logging.info('Loading pretrained ImageNet Swin V2 encoder weights from: %s',
                     pretrained_path)
        report = model.load_pretrained(pretrained_path)
        logging.info(
            'Pretrained loading summary: %d loaded, %d channel-adapted, '
            '%d skipped, %d decoder/MSSD tensors left at random init.',
            len(report['loaded']), len(report['adapted']),
            len(report['skipped']), len(report['not_applicable'])
        )
    else:
        logging.info('Initializing model from scratch.')

    class_distr = CLASS_DISTR.clone()
    if options['agg_to_water']:
        agg_distr = class_distr[-4:].sum()
        class_distr[6] += agg_distr
        class_distr = class_distr[:-4]

    weight = gen_weights(class_distr, c=options['weight_param']).to(device)

    if options['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=weight, gamma=options['focal_gamma'], ignore_index=-1, reduction='mean')
        logging.info("Using Focal Loss with gamma=%.2f", options['focal_gamma'])
    elif options['loss_type'] == 'dice':
        criterion = DiceLoss(num_classes=options['output_channels'], weight=weight, ignore_index=-1)
        logging.info("Using Dice Loss")
    elif options['loss_type'] == 'ce_dice':
        criterion = CEDiceLoss(num_classes=options['output_channels'], weight=weight,
                                ignore_index=-1, dice_weight=options['dice_weight'])
        logging.info("Using CrossEntropy + %.2f * Dice Loss", options['dice_weight'])
    else:
        criterion = ConfidenceWeightedCE(weight=weight, ignore_index=-1)
        logging.info("Using CrossEntropy Loss")

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

    # ---------- Learning-rate schedule: linear warmup + main schedule ----------
    warmup_scheduler, main_scheduler, warmup_steps = build_scheduler(
        optimizer, options, steps_per_epoch=len(train_loader) if options['mode'] == 'train' else 1
    )

    start_epoch = options['resume_from_epoch'] + 1
    epochs = options['epochs']
    eval_every = options['eval_every']

    if options['mode'] == 'train':
        sample_batch = next(iter(train_loader))
        sample_img = sample_batch[0]
        print(f"Image shape: {sample_img.shape}")  # (batch, C, 256, 256); C=11, or 13/17/19 with extra channels
        writer.add_graph(model, sample_img.to(device))

        model.train()
        global_step = 0

        for epoch in range(start_epoch, epochs + 1):
            train_losses = []
            train_samples = 0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc="training")):
                if options['use_confidence_weighting']:
                    images, targets, pixel_weight = batch
                    pixel_weight = pixel_weight.to(device)
                else:
                    images, targets = batch
                    pixel_weight = None
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets, pixel_weight=pixel_weight)
                loss.backward()

                if options['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), options['grad_clip'])

                train_samples += targets.shape[0]
                train_losses.append(loss.item() * targets.shape[0])
                optimizer.step()

                if warmup_scheduler is not None and global_step < warmup_steps:
                    warmup_scheduler.step()

                writer.add_scalar(
                    'training loss', loss.item(),
                    (epoch - 1) * len(train_loader) + batch_idx
                )
                global_step += 1

            avg_train_loss = sum(train_losses) / train_samples
            logging.info("Epoch %d - Training loss: %.4f", epoch, avg_train_loss)

            if epoch % eval_every == 0 or epoch == 1:
                model.eval()
                val_losses = []
                val_samples = 0
                y_true = []
                y_pred = []

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="testing"):
                        if options['use_confidence_weighting']:
                            images, targets, pixel_weight = batch
                            pixel_weight = pixel_weight.to(device)
                        else:
                            images, targets = batch
                            pixel_weight = None
                        images = images.to(device)
                        targets = targets.to(device)

                        logits = model(images)
                        loss = criterion(logits, targets, pixel_weight=pixel_weight)

                        logits = logits.permute(0, 2, 3, 1).reshape(-1, options['output_channels'])
                        targets_flat = targets.reshape(-1)
                        valid_mask = targets_flat != -1
                        logits = logits[valid_mask]
                        targets_flat = targets_flat[valid_mask]

                        probs = F.softmax(logits, dim=1).cpu().numpy()
                        targets_np = targets_flat.cpu().numpy()

                        val_samples += targets_np.shape[0]
                        val_losses.append(loss.item() * targets_np.shape[0])
                        y_pred.extend(probs.argmax(1).tolist())
                        y_true.extend(targets_np.tolist())

                avg_val_loss = sum(val_losses) / val_samples
                metrics = Evaluation(np.array(y_pred), np.array(y_true))

                logging.info(
                    "Epoch %d - Train loss: %.4f - Val loss: %.4f - Val macroF1: %.4f - Val mIoU: %.4f",
                    epoch, avg_train_loss, avg_val_loss, metrics['macroF1'], metrics['IoU']
                )

                # ---- Checkpoint-selection metric ----
                # Weighted CE/focal loss keeps improving on majority classes
                # (Sediment-Laden Water, Clouds, Turbid Water) long after the
                # epoch with the best rare-class performance (Marine Debris,
                # Foam) has passed. Selecting by val loss can therefore save
                # a checkpoint that looks best on the training objective but
                # is not the one that scores best on the metrics actually
                # reported (Macro F1 / mIoU). Default to selecting by
                # Macro F1; --select_metric val_loss restores the old
                # loss-based behaviour for comparison.
                if options['select_metric'] == 'macroF1':
                    current_score = metrics['macroF1']
                elif options['select_metric'] == 'mIoU':
                    current_score = metrics['IoU']
                else:  # 'val_loss' -- higher-is-better via negation, for a uniform comparison below
                    current_score = -avg_val_loss

                if current_score > best_score:
                    best_score = current_score
                    early_stop_counter = 0
                    logging.info(
                        "Best model improved (%s = %.4f, val loss = %.4f)",
                        options['select_metric'],
                        metrics['macroF1'] if options['select_metric'] == 'macroF1'
                        else (metrics['IoU'] if options['select_metric'] == 'mIoU' else avg_val_loss),
                        avg_val_loss,
                    )
                    logging.info("Evaluation after epoch %d: %s", epoch, metrics)

                    save_dir = os.path.join(options['checkpoint_path'], str(epoch))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(model.state_dict(), save_path)
                    logging.info("Best model saved to: %s", save_path)

                    writer.add_scalars(
                        'Loss per epoch',
                        {'Test loss': avg_val_loss, 'Train loss': avg_train_loss},
                        epoch
                    )
                else:
                    early_stop_counter += 1
                    logging.info(
                        "%s did not improve. Early stop counter: %d/%d",
                        options['select_metric'], early_stop_counter, patience
                    )
                    if early_stop_counter >= patience:
                        logging.info('Early stopping triggered after epoch %d.', epoch)
                        print('Early stopping triggered.')
                        break

                writer.add_scalar('Precision/macroPrec', metrics["macroPrec"], epoch)
                writer.add_scalar('Precision/microPrec', metrics["microPrec"], epoch)
                writer.add_scalar('Precision/weightPrec', metrics["weightPrec"], epoch)
                writer.add_scalar('Recall/macroRec', metrics["macroRec"], epoch)
                writer.add_scalar('Recall/microRec', metrics["microRec"], epoch)
                writer.add_scalar('Recall/weightRec', metrics["weightRec"], epoch)
                writer.add_scalar('F1/macroF1', metrics["macroF1"], epoch)
                writer.add_scalar('F1/microF1', metrics["microF1"], epoch)
                writer.add_scalar('F1/weightF1', metrics["weightF1"], epoch)
                writer.add_scalar('IoU/macroIoU', metrics["IoU"], epoch)

                # ---- Step main scheduler once per epoch, after warmup ----
                if global_step >= warmup_steps:
                    if options['scheduler'] == 'plateau':
                        main_scheduler.step(avg_val_loss)
                    else:
                        main_scheduler.step()

                model.train()

    elif options['mode'] == 'test':
        model.eval()
        test_losses = []
        test_samples = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="testing"):
                if options['use_confidence_weighting']:
                    images, targets, pixel_weight = batch
                    pixel_weight = pixel_weight.to(device)
                else:
                    images, targets = batch
                    pixel_weight = None
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                loss = criterion(logits, targets, pixel_weight=pixel_weight)

                logits = logits.permute(0, 2, 3, 1).reshape(-1, options['output_channels'])
                targets_flat = targets.reshape(-1)
                valid_mask = targets_flat != -1
                logits = logits[valid_mask]
                targets_flat = targets_flat[valid_mask]

                probs = F.softmax(logits, dim=1).cpu().numpy()
                targets_np = targets_flat.cpu().numpy()

                test_samples += targets_np.shape[0]
                test_losses.append(loss.item() * targets_np.shape[0])
                y_pred.extend(probs.argmax(1).tolist())
                y_true.extend(targets_np.tolist())

        avg_test_loss = sum(test_losses) / test_samples
        metrics = Evaluation(np.array(y_pred), np.array(y_true))
        logging.info("\nTest loss: %f", avg_test_loss)
        logging.info("STATISTICS: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------
    # Shared options below are kept identical (same flag, same default)
    # to train_unet.py. Only --pretrained_path is specific to this
    # (Swin-UNet V2) script.
    # ------------------------------------------------------------------
    parser.add_argument('--agg_to_water', default=True, type=bool,
                        help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')
    parser.add_argument('--mode', default='train', help="select between 'train' or 'test'")
    parser.add_argument('--variant', default='full', choices=list(VARIANTS.keys()),
                        help="MSSD-Net ablation variant: 'baseline' (plain Swin-UNet V2), "
                             "'+dilated', '+dilated+attention', 'full' (all three), "
                             "or isolated 'only_dilated' / 'only_attention' / 'only_residual'")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to run')
    parser.add_argument('--batch', default=16, type=int, help='Batch size')
    parser.add_argument('--resume_from_epoch', default=0, type=int, help='load model from previous epoch')
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='Path to an ImageNet-pretrained Swin V2 checkpoint '
                             '(swinv2_tiny_patch4_window8_256.pth) to initialise the '
                             'encoder from. If left unset, the checkpoint is '
                             'auto-downloaded to <PROJECT_ROOT>/pretrained/. If set to '
                             'a path that does not exist yet, it is downloaded there. '
                             'Ignored when --use_pretrained false is passed.')
    parser.add_argument('--use_pretrained', default=True, type=bool,
                        help='Initialise the encoder from ImageNet Swin V2 weights '
                             '(auto-downloaded if needed). Set to False to train '
                             'from scratch instead.')
    parser.add_argument('--checkpoint_name', default='best_model.pth', type=str,
                        help='Name of the checkpoint file in the epoch folder (for resume)')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping')
    parser.add_argument('--select_metric', default='macroF1', choices=['macroF1', 'mIoU', 'val_loss'],
                        help="Metric used to pick the checkpoint to save each eval step. "
                             "'macroF1' (default) and 'mIoU' select by validation performance "
                             "on the metrics actually reported, which better tracks rare-class "
                             "quality under MARIDA's class imbalance than 'val_loss'.")

    parser.add_argument('--input_channels', default=11, type=int,
                         help='Number of input bands. Ignored/overridden in practice: the '
                              'model is actually built with in_chans = train_dataset.num_channels '
                              '(11, +6 with --use_spectral_indices, +2 with --use_texture_features), '
                              'so this can never drift out of sync with either flag.')
    parser.add_argument('--use_spectral_indices', default=False, type=bool,
                         help='Append 6 spectral indices (NDVI, NDWI, NDMI, BSI, FAI, FDI) as '
                              'extra input channels after the 11 raw bands -- the same class of '
                              'hand-engineered features that gave MARIDA\'s own Random Forest '
                              'baseline an edge over from-scratch deep models trained on raw '
                              'bands alone. Applies to train/val/test identically (it is a '
                              'feature representation, not a train-only augmentation).')
    parser.add_argument('--use_texture_features', default=False, type=bool,
                         help='Append 2 fast texture features (local std-dev + gradient magnitude '
                              'over a 13x13 window) as extra input channels -- a cheap proxy for '
                              'the GLCM texture features (Contrast, Energy, etc.) that MARIDA\'s '
                              'own feature-importance analysis found to be individually the most '
                              'informative feature for their winning RF variant. Applies to '
                              'train/val/test identically. Can be combined with '
                              '--use_spectral_indices.')
    parser.add_argument('--use_confidence_weighting', default=False, type=bool,
                         help="Weight each pixel's contribution to the loss by its MARIDA "
                              "annotation confidence level (1=high/2=moderate/3=low -> weights "
                              "1, 2/3, 1/3), matching the scheme MARIDA used for their RF "
                              "baseline. REQUIRES a '{name}_conf.tif' confidence raster next to "
                              "each '{name}_cl.tif' mask file -- this naming convention is an "
                              "ASSUMPTION, not verified against a real MARIDA download; if your "
                              "data uses a different layout, this silently falls back to uniform "
                              "weighting for the affected patches (see the startup warning) "
                              "rather than crashing, so double-check the log before trusting "
                              "results. Applies to train/val/test identically.")
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    parser.add_argument('--weight_param', default=1.03, type=float,
                        help='Weighting parameter for Loss Function')

    parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal', 'dice', 'ce_dice'],
                         help="Loss type: 'ce' (weighted cross-entropy, default), 'focal' "
                              "(measurably hurts Macro F1 in our ablation -- kept for "
                              "comparison), 'dice' (soft Dice only), or 'ce_dice' "
                              "(CE + --dice_weight * Dice, generally the recommended one "
                              "to try first: gentler gradient on rare-class pixels than "
                              "Focal, more IoU-aligned than plain CE).")
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Gamma for Focal Loss')
    parser.add_argument('--dice_weight', default=0.5, type=float,
                         help="Weight of the Dice term when --loss_type ce_dice (loss = "
                              "CE + dice_weight * Dice). Ignored for other loss types.")

    parser.add_argument('--oversample_rare', default=True, type=bool,
                         help="Use a WeightedRandomSampler that oversamples training patches "
                              "containing rare classes, instead of plain uniform shuffling. "
                              "Changes what the model actually sees during training, on top "
                              "of (not instead of) loss-based class weighting.")
    parser.add_argument('--rare_classes', default='0,2,3,8', type=str,
                         help="Comma-separated 0-indexed class IDs to oversample, in the "
                              "aggregated 11-class label space used by the masks (as returned "
                              "by GenDEBRIS, i.e. after --agg_to_water). Default '0,2,3,8' "
                              "assumes the confusion-matrix column order from your logs: "
                              "0=Marine Debris, 2=Sparse Sargassum, 3=Natural Organic Material, "
                              "8=Foam. VERIFY this against your actual label mapping before "
                              "relying on it -- if it's wrong, oversampling will boost the "
                              "wrong patches.")
    parser.add_argument('--oversample_boost', default=5.0, type=float,
                         help="Extra sampling weight added per rare class present in a patch "
                              "(on top of a base weight of 1.0). A patch with one rare class "
                              "present is drawn ~(1+boost)x as often as a patch with none; a "
                              "patch with two rare classes present is drawn ~(1+2*boost)x as "
                              "often, and so on. Higher = more aggressive oversampling.")

    parser.add_argument('--copy_paste_prob', default=0.0, type=float,
                         help="Probability, per training patch, of pasting rare-class pixels "
                              "(--rare_classes) from a randomly chosen donor patch onto the "
                              "current one at the same spatial positions -- image bands and "
                              "mask label are copied together, so they stay consistent. "
                              "Multiplies rare-class training signal on top of (not instead "
                              "of) --oversample_rare. 0 (default) disables it. Train split "
                              "only; val/test are never augmented this way.")
    parser.add_argument('--spectral_jitter_prob', default=0.0, type=float,
                         help="Probability, per training patch, of multiplying each Sentinel-2 "
                              "band by an independent random factor close to 1.0 (see "
                              "--spectral_jitter_strength), simulating sensor/atmospheric "
                              "variation. 0 (default) disables it. Train split only.")
    parser.add_argument('--spectral_jitter_strength', default=0.05, type=float,
                         help="Each band's spectral-jitter factor is drawn uniformly from "
                              "[1 - strength, 1 + strength]. Only used when "
                              "--spectral_jitter_prob > 0.")

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--encoder_lr_mult', default=1.0, type=float,
                         help="Multiplier applied to --lr for the ImageNet-pretrained encoder "
                              "params (patch_embed + layers.*); everything else (decoder, MSSD "
                              "modules, and the randomly-initialized relative-position-bias "
                              "tensors) uses --lr directly. Default 1.0 = single shared LR "
                              "(previous behavior, unchanged). Try e.g. 0.1 so the pretrained "
                              "ImageNet features move slower than the from-scratch parts.")
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--scheduler', default='sgdr',
                        choices=['sgdr', 'plateau', 'multistep', 'cosine'],
                        help='Learning rate scheduler type (sgdr = cosine annealing with warm restarts)')
    parser.add_argument('--lr_steps', default='[40]', type=str, help='Steps for multistep scheduler')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of warmup epochs')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='Gradient clipping value (0 = no clip)')

    parser.add_argument('--train_rotations', default='[-90,0,90,180]', type=str,
                        help='Random-rotation augmentation angles for training, as a '
                             'Python list literal (each must be a multiple of 90, since '
                             'GenDEBRIS rounds the rotated mask back to integer class '
                             'labels). Pass "[]" or "[0]" to disable rotation augmentation.')
    parser.add_argument('--train_hflip', default=True, type=bool,
                        help='Apply random horizontal flip augmentation during training.')

    parser.add_argument('--checkpoint_path', default=os.path.join(PROJECT_ROOT, 'trained_models'),
                        help='base folder to save checkpoints into (variant name is appended automatically)')
    parser.add_argument('--eval_every', default=1, type=int, help='How frequently to run evaluation (epochs)')

    parser.add_argument('--num_workers', default=4, type=int,
                        help='How many cpus for loading data (0 is the main process)')
    parser.add_argument('--pin_memory', default=True, type=bool, help='Use pinned memory or not')
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Number of samples loaded in advance by each worker')
    parser.add_argument('--persistent_workers', default=True, type=bool,
                        help='Keep worker Dataset instances alive between epochs')
    parser.add_argument('--tensorboard', default='tsboard_swin', type=str,
                        help='base name for tensorboard run (variant name is appended automatically)')

    args = parser.parse_args()
    opts = vars(args)

    if opts['scheduler'] == 'multistep':
        lr_steps = ast.literal_eval(opts['lr_steps'])
        if isinstance(lr_steps, int):
            lr_steps = [lr_steps]
        opts['lr_steps'] = lr_steps
    else:
        opts['lr_steps'] = []

    train_rotations = ast.literal_eval(opts['train_rotations'])
    if isinstance(train_rotations, int):
        train_rotations = [train_rotations]
    for angle in train_rotations:
        if angle % 90 != 0:
            raise ValueError(
                f"--train_rotations angles must be multiples of 90 (GenDEBRIS rounds "
                f"the rotated mask back to integer class labels); got {angle}"
            )
    opts['train_rotations'] = train_rotations

    # Namespace checkpoints and tensorboard logs by variant, so running the
    # ablation study (baseline, +dilated, +dilated+attention, full, ...)
    # never overwrites a previous run's checkpoints or logs.
    opts['checkpoint_path'] = os.path.join(opts['checkpoint_path'], opts['variant'])
    opts['tensorboard'] = f"{opts['tensorboard']}_{opts['variant']}"

    logging.info('parsed input parameters:\n%s', json.dumps(opts, indent=2))
    main(opts)