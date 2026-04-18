# -*- coding: utf-8 -*-
"""
train.py  –  Unified training script for the MSSD Framework.

Supports two model families selectable via --model:
  unet      → UNet with pluggable CNN backbones (none | resnet18 | mobilenetv2 | efficientnetv2)
  swin_unet → Swin‑UNet (Vision‑Transformer backbone)

Usage examples
--------------
# Train UNet (plain, no backbone)
python train.py --model unet --backbone none --mode train

# Train UNet with ResNet‑18 encoder
python train.py --model unet --backbone resnet18 --mode train

# Train Swin‑UNet
python train.py --model swin_unet --mode train --lr 5e-5 --warmup_epochs 5

# Test a saved checkpoint
python train.py --model swin_unet --mode test --resume_from_epoch 42
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
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup  (keep identical to the original convention)
# ---------------------------------------------------------------------------
PROJECT_ROOT = up(up(up(os.path.abspath(__file__))))   # repo root
SCRIPT_DIR   = up(os.path.abspath(__file__))           # directory of this file

sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

# ---------------------------------------------------------------------------
# Logging  –  file + console
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'logs', 'train.log'),
    filemode='a',
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(_console)

logging.info('*' * 10)


# ===========================================================================
# Reproducibility helpers
# ===========================================================================

def seed_all(seed: int = 0) -> None:
    """Fix all random seeds for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """DataLoader worker initialisation."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===========================================================================
# Loss functions
# ===========================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-class segmentation with ignore_index support."""

    def __init__(self, alpha=None, gamma: float = 2.0,
                 ignore_index: int = -1, reduction: str = 'mean'):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.reduction    = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none',
                                     weight=self.alpha, ignore_index=self.ignore_index)
        valid_mask = targets != self.ignore_index
        pt         = torch.exp(-ce_loss)
        focal      = ((1 - pt) ** self.gamma) * ce_loss * valid_mask

        if self.reduction == 'mean':
            return focal.sum() / valid_mask.sum() if valid_mask.sum() > 0 \
                   else torch.tensor(0.0, device=inputs.device)
        if self.reduction == 'sum':
            return focal.sum()
        return focal


def build_criterion(options: dict, weight: torch.Tensor) -> nn.Module:
    """Instantiate the loss function from CLI options."""
    if options['loss_type'] == 'focal':
        logging.info("Using Focal Loss  gamma=%.2f", options['focal_gamma'])
        return FocalLoss(alpha=weight, gamma=options['focal_gamma'],
                         ignore_index=-1, reduction='mean')
    logging.info("Using CrossEntropy Loss")
    return nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=weight)


# ===========================================================================
# Model factory
# ===========================================================================

def build_model(options: dict) -> nn.Module:
    """Return the requested model instance."""
    model_type = options['model']

    if model_type == 'unet':
        from unet import UNet  # local import keeps top‑level imports clean
        model = UNet(
            input_bands=options['input_channels'],
            output_classes=options['output_channels'],
            hidden_channels=options['hidden_channels'],
            backbone=options['backbone'],
        )
        logging.info("Model: UNet  backbone=%s  hidden=%d",
                     options['backbone'], options['hidden_channels'])

    elif model_type == 'swin_unet':
        from utils.config import Config
        from vision_transformer import SwinUnet
        config = Config()
        model  = SwinUnet(
            config,
            img_size=config.DATA.IMG_SIZE,
            num_classes=options['output_channels'],
        )
        logging.info("Model: Swin‑UNet  img_size=%d  classes=%d",
                     config.DATA.IMG_SIZE, options['output_channels'])

    else:
        raise ValueError(f"Unknown --model value: '{model_type}'. "
                         "Choose from: unet, swin_unet")

    return model


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def load_checkpoint(model: nn.Module, options: dict,
                    device: torch.device) -> None:
    """Load weights into model in‑place (resume or pretrained)."""
    if options['resume_from_epoch'] > 1:
        ckpt_dir  = os.path.join(options['checkpoint_path'],
                                 str(options['resume_from_epoch']))
        ckpt_file = os.path.join(ckpt_dir, options['checkpoint_name'])
        logging.info("Resuming from epoch %d  file=%s",
                     options['resume_from_epoch'], ckpt_file)
        state = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state)
        logging.info("Checkpoint loaded.")
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif options.get('pretrained_path'):
        path = options['pretrained_path']
        logging.info("Loading pretrained weights from: %s", path)
        if options['model'] == 'swin_unet' and hasattr(model, 'load_pretrained'):
            model.load_pretrained(path)
        else:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state, strict=False)
            logging.info("Pretrained weights loaded (non‑strict).")

    else:
        logging.info("Initialising model from scratch.")


# ===========================================================================
# Scheduler factory
# ===========================================================================

def build_scheduler(optimizer, options: dict, steps_per_epoch: int):
    """
    Return (scheduler, plateau_scheduler).

    plateau_scheduler is only non‑None when warmup is active AND
    scheduler='plateau', because ReduceLROnPlateau cannot be wrapped
    in SequentialLR.
    """
    warmup_steps    = options.get('warmup_epochs', 0) * steps_per_epoch
    total_steps     = options['epochs'] * steps_per_epoch
    sched_type      = options['scheduler']
    plateau_sched   = None

    def _make_main_scheduler():
        if sched_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=1e-6,
            )
        if sched_type == 'multistep':
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, options['lr_steps'], gamma=0.1,
            )
        return None  # plateau handled separately

    if warmup_steps > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.01,
                                end_factor=1.0, total_iters=warmup_steps)
        if sched_type == 'plateau':
            # Warmup is stepped batch‑wise; plateau is stepped epoch‑wise
            scheduler      = warmup_sched
            plateau_sched  = ReduceLROnPlateau(optimizer, mode='min',
                                               factor=0.1, patience=10)
        else:
            main = _make_main_scheduler()
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main],
                milestones=[warmup_steps],
            ) if main else warmup_sched

    else:  # no warmup
        if sched_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                          factor=0.1, patience=10)
        else:
            scheduler = _make_main_scheduler()

    logging.info("Scheduler: %s  warmup_epochs=%d",
                 sched_type, options.get('warmup_epochs', 0))
    return scheduler, plateau_sched


# ===========================================================================
# Data helpers
# ===========================================================================

def build_dataloaders(options: dict, g: torch.Generator):
    """Return (train_loader, val_or_test_loader) depending on mode."""
    from dataloader import (GenDEBRIS, bands_mean, bands_std,
                            RandomRotationTransform, class_distr, gen_weights)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        RandomRotationTransform([-90, 0, 90, 180]),
        transforms.RandomHorizontalFlip(),
    ])
    transform_eval  = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)

    loader_kwargs = dict(
        batch_size=options['batch'],
        num_workers=options['num_workers'],
        pin_memory=options['pin_memory'],
        prefetch_factor=options['prefetch_factor'],
        persistent_workers=options['persistent_workers'],
        worker_init_fn=seed_worker,
        generator=g,
    )

    if options['mode'] == 'train':
        train_ds = GenDEBRIS('train', transform=transform_train,
                             standardization=standardization,
                             agg_to_water=options['agg_to_water'])
        val_ds   = GenDEBRIS('val', transform=transform_eval,
                             standardization=standardization,
                             agg_to_water=options['agg_to_water'])
        train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
        val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
        return train_loader, val_loader, class_distr, gen_weights

    elif options['mode'] == 'test':
        test_ds = GenDEBRIS('test', transform=transform_eval,
                            standardization=standardization,
                            agg_to_water=options['agg_to_water'])
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        return None, test_loader, class_distr, gen_weights

    raise ValueError("--mode must be 'train' or 'test'")


def prepare_class_weights(class_distr, options: dict,
                          device: torch.device) -> torch.Tensor:
    """Optionally aggregate water classes, then compute loss weights."""
    distr = class_distr.clone() if hasattr(class_distr, 'clone') \
            else list(class_distr)

    if options['agg_to_water']:
        agg = sum(distr[-4:]) if isinstance(distr, list) else distr[-4:].sum()
        distr[6] += agg
        distr = distr[:-4]

    from dataloader import gen_weights
    return gen_weights(distr, c=options['weight_param']).to(device)


# ===========================================================================
# Pixel‑level evaluation helper (shared by both model branches)
# ===========================================================================

def flatten_logits_targets(logits: torch.Tensor, targets: torch.Tensor,
                            n_classes: int):
    """Return (logits_flat, targets_flat) with ignore‑index pixels removed."""
    logits_flat  = logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
    targets_flat = targets.reshape(-1)
    mask         = targets_flat != -1
    return logits_flat[mask], targets_flat[mask]


# ===========================================================================
# Train / validation loops
# ===========================================================================

def run_train_epoch(model, loader, criterion, optimizer,
                    scheduler, plateau_scheduler,
                    writer, epoch, options, global_step, warmup_steps):
    """Single training epoch. Returns (avg_loss, global_step)."""
    model.train()
    total_loss   = 0.0
    total_pixels = 0

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="train")):
        images  = images.to(options['device'])
        targets = targets.to(options['device'])

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, targets)
        loss.backward()

        if options.get('grad_clip', 0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), options['grad_clip'])

        optimizer.step()

        n_pixels     = targets.shape[0]
        total_pixels += n_pixels
        total_loss   += loss.item() * n_pixels

        # Batch‑level warmup stepping (Swin‑UNet style)
        if warmup_steps > 0 and global_step < warmup_steps:
            scheduler.step()

        writer.add_scalar('Loss/train_batch', loss.item(),
                          (epoch - 1) * len(loader) + batch_idx)
        global_step += 1

    return total_loss / total_pixels, global_step


def run_eval_epoch(model, loader, criterion, options):
    """Validation / test loop. Returns (avg_loss, y_pred, y_true)."""
    model.eval()
    total_loss   = 0.0
    total_pixels = 0
    y_pred       = []
    y_true       = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval"):
            images  = images.to(options['device'])
            targets = targets.to(options['device'])

            logits = model(images)
            loss   = criterion(logits, targets)

            logits_flat, targets_flat = flatten_logits_targets(
                logits, targets, options['output_channels']
            )

            probs        = F.softmax(logits_flat, dim=1).cpu().numpy()
            targets_np   = targets_flat.cpu().numpy()
            n_pixels     = targets_np.shape[0]

            total_pixels += n_pixels
            total_loss   += loss.item() * n_pixels
            y_pred.extend(probs.argmax(1).tolist())
            y_true.extend(targets_np.tolist())

    return total_loss / total_pixels, np.array(y_pred), np.array(y_true)


# ===========================================================================
# Main entry point
# ===========================================================================

def main(options: dict) -> None:
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)

    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options['device'] = device
    logging.info("Using device: %s", device)

    # ----- TensorBoard -----
    writer = SummaryWriter(os.path.join(PROJECT_ROOT, 'logs',
                                        options['tensorboard']))

    # ----- Data -----
    train_loader, eval_loader, class_distr, gen_weights_fn = \
        build_dataloaders(options, g)

    # ----- Model -----
    model = build_model(options)
    model.to(device)
    load_checkpoint(model, options, device)

    # ----- Loss -----
    weight    = prepare_class_weights(class_distr, options, device)
    criterion = build_criterion(options, weight)

    # ----- Metrics imports -----
    from utils.metrics import Evaluation, confusion_matrix
    from utils.assets  import labels as all_labels
    labels = list(all_labels)
    if options['agg_to_water']:
        labels = labels[:-4]

    # ===================================================================
    # TRAINING MODE
    # ===================================================================
    if options['mode'] == 'train':

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=options['lr'],
            weight_decay=options['decay'],
        )

        steps_per_epoch = len(train_loader)
        scheduler, plateau_scheduler = build_scheduler(
            optimizer, options, steps_per_epoch
        )
        warmup_steps = options.get('warmup_epochs', 0) * steps_per_epoch

        # Log graph once
        sample_img, _ = next(iter(train_loader))
        writer.add_graph(model, sample_img.to(device))

        best_loss          = float('inf')
        early_stop_counter = 0
        global_step        = 0
        start_epoch        = options['resume_from_epoch'] + 1

        for epoch in range(start_epoch, options['epochs'] + 1):

            # -- Train --
            avg_train_loss, global_step = run_train_epoch(
                model, train_loader, criterion, optimizer,
                scheduler, plateau_scheduler,
                writer, epoch, options, global_step, warmup_steps,
            )
            logging.info("Epoch %d  train_loss=%.4f", epoch, avg_train_loss)

            # -- Validate --
            if epoch % options['eval_every'] == 0 or epoch == 1:
                avg_val_loss, y_pred, y_true = run_eval_epoch(
                    model, eval_loader, criterion, options
                )

                acc      = Evaluation(y_pred, y_true)
                conf_mat = confusion_matrix(y_true, y_pred, labels)
                logging.info("Epoch %d  val_loss=%.4f", epoch, avg_val_loss)
                logging.info("Confusion Matrix:\n%s", conf_mat.to_string())

                # -- Save best model --
                if avg_val_loss < best_loss:
                    best_loss          = avg_val_loss
                    early_stop_counter = 0
                    logging.info("Best val_loss improved → %.4f", best_loss)
                    logging.info("Metrics: %s", acc)

                    save_dir = os.path.join(options['checkpoint_path'],
                                            str(epoch))
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, 'best_model.pth'))
                    logging.info("Saved → %s", save_dir)

                    writer.add_scalars('Loss/epoch',
                                       {'val': avg_val_loss,
                                        'train': avg_train_loss}, epoch)
                else:
                    early_stop_counter += 1
                    logging.info("No improvement. EarlyStop %d/%d",
                                 early_stop_counter, options['patience'])
                    if early_stop_counter >= options['patience']:
                        logging.info("Early stopping at epoch %d.", epoch)
                        break

                # TensorBoard metrics
                for prefix, key in [
                    ('Precision/macro',  'macroPrec'),
                    ('Precision/micro',  'microPrec'),
                    ('Precision/weight', 'weightPrec'),
                    ('Recall/macro',     'macroRec'),
                    ('Recall/micro',     'microRec'),
                    ('Recall/weight',    'weightRec'),
                    ('F1/macro',         'macroF1'),
                    ('F1/micro',         'microF1'),
                    ('F1/weight',        'weightF1'),
                    ('IoU/macro',        'IoU'),
                ]:
                    writer.add_scalar(prefix, acc[key], epoch)

                # -- Step scheduler (epoch‑level) --
                if options['scheduler'] == 'plateau':
                    if warmup_steps > 0 and global_step < warmup_steps:
                        pass  # still in batch‑level warmup
                    elif plateau_scheduler is not None:
                        plateau_scheduler.step(avg_val_loss)
                    else:
                        scheduler.step(avg_val_loss)
                elif warmup_steps == 0:
                    scheduler.step()

    # ===================================================================
    # TEST MODE
    # ===================================================================
    elif options['mode'] == 'test':
        avg_test_loss, y_pred, y_true = run_eval_epoch(
            model, eval_loader, criterion, options
        )
        acc      = Evaluation(y_pred, y_true)
        conf_mat = confusion_matrix(y_true, y_pred, labels)
        logging.info("Test loss: %.4f", avg_test_loss)
        logging.info("Metrics:\n%s", acc)
        logging.info("Confusion Matrix:\n%s", conf_mat.to_string())
        print(f"Test loss: {avg_test_loss:.4f}")
        print("Evaluation:", acc)
        print("Confusion Matrix:\n", conf_mat)

    else:
        raise ValueError("--mode must be 'train' or 'test'")

    writer.close()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description='MSSD Framework – unified training script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Model selection ----
    model_grp = parser.add_argument_group('Model')
    model_grp.add_argument(
        '--model', default='unet',
        choices=['unet', 'swin_unet'],
        help='Model architecture to train',
    )
    model_grp.add_argument(
        '--backbone', default='none',
        choices=['none', 'resnet18', 'mobilenetv2', 'efficientnetv2'],
        help='CNN backbone for UNet (ignored when --model swin_unet)',
    )
    model_grp.add_argument(
        '--hidden_channels', default=16, type=int,
        help='Hidden feature size (UNet only)',
    )
    model_grp.add_argument(
        '--input_channels', default=11, type=int,
        help='Number of input spectral bands',
    )
    model_grp.add_argument(
        '--output_channels', default=11, type=int,
        help='Number of output segmentation classes',
    )

    # ---- Training mode ----
    run_grp = parser.add_argument_group('Run')
    run_grp.add_argument(
        '--mode', default='train', choices=['train', 'test'],
        help='Run mode',
    )
    run_grp.add_argument('--epochs', default=150, type=int)
    run_grp.add_argument('--batch',  default=4,   type=int)
    run_grp.add_argument(
        '--resume_from_epoch', default=0, type=int,
        help='Resume training from this epoch (0 = from scratch)',
    )
    run_grp.add_argument(
        '--pretrained_path', default=None, type=str,
        help='Path to pretrained weights',
    )
    run_grp.add_argument(
        '--checkpoint_name', default='best_model.pth', type=str,
    )
    run_grp.add_argument(
        '--patience', default=10, type=int,
        help='Early‑stopping patience (epochs without improvement)',
    )
    run_grp.add_argument(
        '--agg_to_water', default=True, type=bool,
        help='Aggregate water‑adjacent classes into Marine Water',
    )

    # ---- Loss ----
    loss_grp = parser.add_argument_group('Loss')
    loss_grp.add_argument(
        '--loss_type', default='ce', choices=['ce', 'focal'],
    )
    loss_grp.add_argument(
        '--focal_gamma', default=2.0, type=float,
    )
    loss_grp.add_argument(
        '--weight_param', default=1.03, type=float,
        help='Class‑weight exponent c  (gen_weights)',
    )

    # ---- Optimiser / scheduler ----
    opt_grp = parser.add_argument_group('Optimiser')
    opt_grp.add_argument('--lr',    default=1e-4, type=float)
    opt_grp.add_argument('--decay', default=0.0,  type=float,
                         help='Weight decay (L2 regularisation)')
    opt_grp.add_argument(
        '--scheduler', default='plateau',
        choices=['plateau', 'multistep', 'cosine'],
    )
    opt_grp.add_argument(
        '--lr_steps', default='[40]', type=str,
        help='Epoch milestones for MultiStepLR, e.g. "[40,80]"',
    )
    opt_grp.add_argument(
        '--warmup_epochs', default=0, type=int,
        help='Linear warmup epochs (useful for Swin‑UNet)',
    )
    opt_grp.add_argument(
        '--grad_clip', default=0.0, type=float,
        help='Gradient clipping max‑norm (0 = disabled)',
    )

    # ---- I/O ----
    io_grp = parser.add_argument_group('I/O')
    io_grp.add_argument(
        '--checkpoint_path',
        default=os.path.join(SCRIPT_DIR, 'trained_models'),
    )
    io_grp.add_argument(
        '--eval_every', default=1, type=int,
        help='Evaluate on validation set every N epochs',
    )
    io_grp.add_argument(
        '--tensorboard', default='tsboard_segm', type=str,
    )

    # ---- DataLoader ----
    dl_grp = parser.add_argument_group('DataLoader')
    dl_grp.add_argument('--num_workers',       default=1,    type=int)
    dl_grp.add_argument('--pin_memory',         default=False, type=bool)
    dl_grp.add_argument('--prefetch_factor',    default=1,    type=int)
    dl_grp.add_argument('--persistent_workers', default=True, type=bool)

    args    = parser.parse_args()
    options = vars(args)

    # Parse lr_steps string → list
    lr_steps = ast.literal_eval(options['lr_steps'])
    if isinstance(lr_steps, int):
        lr_steps = [lr_steps]
    elif not isinstance(lr_steps, list):
        raise ValueError("--lr_steps must be an int or a list, e.g. '[40,80]'")
    options['lr_steps'] = lr_steps

    logging.info("Parsed options:\n%s", json.dumps(
        {k: v for k, v in options.items() if k != 'device'}, indent=2
    ))
    return options


if __name__ == '__main__':
    main(parse_args())