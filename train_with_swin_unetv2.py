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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = up(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.config import Config
from vision_transformer import SwinUnet
from dataloader import (
    GenDEBRIS, BANDS_MEAN, BANDS_STD,
    RandomRotationTransform, gen_weights, CLASS_DISTR
)

sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))
from utils.metrics import Evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logging.info('*' * 10)


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


class FocalLoss(nn.Module):
    """Focal Loss for multi-class segmentation."""

    def __init__(self, alpha=None, gamma=2, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none',
            weight=self.alpha, ignore_index=self.ignore_index
        )
        valid_mask = targets != self.ignore_index
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        focal_loss = focal_loss * valid_mask

        if self.reduction == 'mean':
            if valid_mask.sum() > 0:
                return focal_loss.sum() / valid_mask.sum()
            return torch.tensor(0.0, device=inputs.device)
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


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

    best_loss = float('inf')
    early_stop_counter = 0
    patience = options['patience']
    writer = SummaryWriter(os.path.join(PROJECT_ROOT, 'logs', options['tensorboard']))

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        RandomRotationTransform([-90, 0, 90, 180]),
        transforms.RandomHorizontalFlip()
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)

    if options['mode'] == 'train':
        train_dataset = GenDEBRIS(
            'train', transform=transform_train, standardization=standardization,
            agg_to_water=options['agg_to_water']
        )
        val_dataset = GenDEBRIS(
            'val', transform=transform_test, standardization=standardization,
            agg_to_water=options['agg_to_water']
        )

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
            agg_to_water=options['agg_to_water']
        )
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

    # Model
    config = Config()
    model = SwinUnet(
        config, img_size=config.DATA.IMG_SIZE,
        num_classes=options['output_channels']
    )
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
    elif options.get('pretrained_path'):
        model.load_pretrained(options['pretrained_path'])
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
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=weight)
        logging.info("Using CrossEntropy Loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['decay'])

    # ---------- Learning-rate schedule: linear warmup + main schedule ----------
    warmup_scheduler, main_scheduler, warmup_steps = build_scheduler(
        optimizer, options, steps_per_epoch=len(train_loader) if options['mode'] == 'train' else 1
    )

    start_epoch = options['resume_from_epoch'] + 1
    epochs = options['epochs']
    eval_every = options['eval_every']

    if options['mode'] == 'train':
        sample_img, _ = next(iter(train_loader))
        print(f"Image shape: {sample_img.shape}")  # (batch, 11, 256, 256)
        writer.add_graph(model, sample_img.to(device))

        model.train()
        global_step = 0

        for epoch in range(start_epoch, epochs + 1):
            train_losses = []
            train_samples = 0

            for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="training")):
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
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
                    for images, targets in tqdm(val_loader, desc="testing"):
                        images = images.to(device)
                        targets = targets.to(device)

                        logits = model(images)
                        loss = criterion(logits, targets)

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
                    "Epoch %d - Train loss: %.4f - Val loss: %.4f",
                    epoch, avg_train_loss, avg_val_loss
                )

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    early_stop_counter = 0
                    logging.info("Best validation loss improved to: %.4f", best_loss)
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
                        "Validation loss did not improve. Early stop counter: %d/%d",
                        early_stop_counter, patience
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
            for images, targets in tqdm(test_loader, desc="testing"):
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                loss = criterion(logits, targets)

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
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs to run')
    parser.add_argument('--batch', default=16, type=int, help='Batch size')
    parser.add_argument('--resume_from_epoch', default=0, type=int, help='load model from previous epoch')
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='Path to pre-trained weights (optional)')
    parser.add_argument('--checkpoint_name', default='best_model.pth', type=str,
                        help='Name of the checkpoint file in the epoch folder (for resume)')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping')

    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    parser.add_argument('--weight_param', default=1.03, type=float,
                        help='Weighting parameter for Loss Function')

    parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal'], help='Loss type')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Gamma for Focal Loss')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--scheduler', default='sgdr',
                        choices=['sgdr', 'plateau', 'multistep', 'cosine'],
                        help='Learning rate scheduler type (sgdr = cosine annealing with warm restarts)')
    parser.add_argument('--lr_steps', default='[40]', type=str, help='Steps for multistep scheduler')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of warmup epochs')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='Gradient clipping value (0 = no clip)')

    parser.add_argument('--checkpoint_path', default=os.path.join(PROJECT_ROOT, 'trained_models'),
                        help='folder to save checkpoints into')
    parser.add_argument('--eval_every', default=1, type=int, help='How frequently to run evaluation (epochs)')

    parser.add_argument('--num_workers', default=4, type=int,
                        help='How many cpus for loading data (0 is the main process)')
    parser.add_argument('--pin_memory', default=True, type=bool, help='Use pinned memory or not')
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Number of samples loaded in advance by each worker')
    parser.add_argument('--persistent_workers', default=True, type=bool,
                        help='Keep worker Dataset instances alive between epochs')
    parser.add_argument('--tensorboard', default='tsboard_swin', type=str, help='Name for tensorboard run')

    args = parser.parse_args()
    opts = vars(args)

    if opts['scheduler'] == 'multistep':
        lr_steps = ast.literal_eval(opts['lr_steps'])
        if isinstance(lr_steps, int):
            lr_steps = [lr_steps]
        opts['lr_steps'] = lr_steps
    else:
        opts['lr_steps'] = []

    logging.info('parsed input parameters:\n%s', json.dumps(opts, indent=2))
    main(opts)
