# -*- coding: utf-8 -*-
"""
Evaluation script for Swin‑UNet on the test set.
"""

import argparse
import logging
import os
import random
import sys
from os.path import dirname as up

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = up(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.config import Config
from vision_transformer import SwinUnet
from dataloader import GenDEBRIS, BANDS_MEAN, BANDS_STD

sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))
from utils.metrics import Evaluation, confusion_matrix
from utils.assets import labels

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

logging.basicConfig(
    filename=os.path.join(PROJECT_ROOT, 'logs', 'evaluating_swin.log'),
    filemode='a', level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logging.info('*' * 10)


def evaluate_model(model_path, options, device, test_loader, labels_subset):
    """Load a model and evaluate on test set."""
    config = Config()
    model = SwinUnet(
        config, img_size=config.DATA.IMG_SIZE,
        num_classes=options['output_channels']
    )
    model.to(device)

    logging.info('Loading model for evaluation from: %s', model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    logging.info('Model loaded successfully.')
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="testing"):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            logits = logits.permute(0, 2, 3, 1).reshape(-1, options['output_channels'])
            targets_flat = targets.reshape(-1)
            valid_mask = targets_flat != -1
            logits = logits[valid_mask]
            targets_flat = targets_flat[valid_mask]

            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            targets_np = targets_flat.cpu().numpy()

            y_pred.extend(probs.argmax(1).tolist())
            y_true.extend(targets_np.tolist())

    acc = Evaluation(y_pred, y_true)
    conf_mat = confusion_matrix(y_true, y_pred, labels_subset)
    return acc, conf_mat


def main(options):
    """Main evaluation loop."""
    # Transforms – keep original size
    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)

    test_dataset = GenDEBRIS(
        'test',
        transform=transform_test,
        standardization=standardization,
        agg_to_water=options['agg_to_water']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=options['batch'],
        shuffle=False,
        num_workers=options['num_workers'],
        pin_memory=True
    )

    # Adjust labels if aggregation is applied
    global labels
    if options['agg_to_water']:
        labels_subset = labels[:-4]
    else:
        labels_subset = labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if options['epochs_list']:
        # Evaluate multiple checkpoints
        epochs = [int(e) for e in options['epochs_list'].split(',')]
        for ep in epochs:
            model_path = os.path.join(
                options['checkpoint_path'], str(ep), options['checkpoint_name']
            )
            if not os.path.exists(model_path):
                logging.warning("Checkpoint not found: %s", model_path)
                continue
            acc, conf_mat = evaluate_model(
                model_path, options, device, test_loader, labels_subset
            )
            logging.info("\nResults for epoch %d:", ep)
            logging.info("Evaluation: %s", acc)
            logging.info("Confusion Matrix:\n%s", conf_mat.to_string())
            print(f"\nEpoch {ep}:")
            print("Evaluation:", acc)
            print("Confusion Matrix:\n", conf_mat)
    else:
        # Single model evaluation
        model_path = options['model_path']
        acc, conf_mat = evaluate_model(
            model_path, options, device, test_loader, labels_subset
        )
        logging.info("\nSTATISTICS: \n%s", acc)
        logging.info("Confusion Matrix:\n%s", conf_mat.to_string())
        print("Evaluation:", acc)
        print("Confusion Matrix:\n", conf_mat)

    # Optionally generate prediction masks (placeholder)
    if options['predict_masks']:
        # Add mask generation code here if needed
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--agg_to_water', default=True, type=bool,
        help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water'
    )
    parser.add_argument(
        '--batch', default=8, type=int, help='Batch size for testing'
    )

    parser.add_argument(
        '--input_channels', default=11, type=int, help='Number of input bands'
    )
    parser.add_argument(
        '--output_channels', default=11, type=int, help='Number of output classes'
    )

    parser.add_argument(
        '--model_path',
        default=os.path.join(PROJECT_ROOT, 'trained_models', 'best', 'best_model.pth'),
        help='Path to SwinUnet PyTorch model'
    )
    parser.add_argument(
        '--epochs_list', default=None, type=str,
        help='Comma‑separated list of epochs to evaluate (e.g., "50,75,100")'
    )
    parser.add_argument(
        '--checkpoint_path',
        default=os.path.join(PROJECT_ROOT, 'trained_models'),
        help='Base folder where checkpoints are saved (used with epochs_list)'
    )
    parser.add_argument(
        '--checkpoint_name', default='best_model.pth', type=str,
        help='Name of checkpoint file inside each epoch folder'
    )

    parser.add_argument(
        '--predict_masks', default=False, type=bool,
        help='Generate test set prediction masks?'
    )
    parser.add_argument(
        '--gen_masks_path',
        default=os.path.join(PROJECT_ROOT, 'data', 'predicted_swin'),
        help='Path to where to store predictions'
    )

    parser.add_argument(
        '--num_workers', default=4, type=int,
        help='Number of data loading workers'
    )

    args = parser.parse_args()
    opts = vars(args)
    main(opts)