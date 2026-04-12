# -*- coding: utf-8 -*-
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(up(os.path.abspath(__file__)))
from unet import UNet
from dataloader import GenDEBRIS, bands_mean, bands_std, RandomRotationTransform, class_distr, gen_weights

sys.path.append(os.path.join(up(up(up(os.path.abspath(__file__)))), 'utils'))
from utils.metrics import Evaluation, confusion_matrix
from utils.assets import labels

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(filename=os.path.join(root_path, 'logs', 'log_unet.log'), filemode='a', level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')
# Ajout d'un handler console pour voir les logs en direct
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info('*' * 10)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with ignore_index support.
    """
    def __init__(self, alpha=None, gamma=2, ignore_index=-1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, ignore_index=self.ignore_index)
        valid_mask = targets != self.ignore_index
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        focal_loss = focal_loss * valid_mask
        if self.reduction == 'mean':
            if valid_mask.sum() > 0:
                return focal_loss.sum() / valid_mask.sum()
            else:
                return torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


###############################################################
# Training                                                    #
###############################################################

def main(options):
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)
    best_loss_value = float('inf')
    early_stop_counter = 0
    patience = options['patience']
    writer = SummaryWriter(os.path.join(root_path, 'logs', options['tensorboard']))

    # Transformations
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          RandomRotationTransform([-90, 0, 90, 180]),
                                          transforms.RandomHorizontalFlip()])

    transform_test = transforms.Compose([transforms.ToTensor()])

    standardization = transforms.Normalize(bands_mean, bands_std)

    if options['mode'] == 'train':
        dataset_train = GenDEBRIS('train', transform=transform_train, standardization=standardization,
                                  agg_to_water=options['agg_to_water'])
        dataset_test = GenDEBRIS('val', transform=transform_test, standardization=standardization,
                                 agg_to_water=options['agg_to_water'])

        train_loader = DataLoader(dataset_train,
                                  batch_size=options['batch'],
                                  shuffle=True,
                                  num_workers=options['num_workers'],
                                  pin_memory=options['pin_memory'],
                                  prefetch_factor=options['prefetch_factor'],
                                  persistent_workers=options['persistent_workers'],
                                  worker_init_fn=seed_worker,
                                  generator=g)

        test_loader = DataLoader(dataset_test,
                                 batch_size=options['batch'],
                                 shuffle=False,
                                 num_workers=options['num_workers'],
                                 pin_memory=options['pin_memory'],
                                 prefetch_factor=options['prefetch_factor'],
                                 persistent_workers=options['persistent_workers'],
                                 worker_init_fn=seed_worker,
                                 generator=g)

    elif options['mode'] == 'test':
        dataset_test = GenDEBRIS('test', transform=transform_test, standardization=standardization,
                                 agg_to_water=options['agg_to_water'])

        test_loader = DataLoader(dataset_test,
                                 batch_size=options['batch'],
                                 shuffle=False,
                                 num_workers=options['num_workers'],
                                 pin_memory=options['pin_memory'],
                                 prefetch_factor=options['prefetch_factor'],
                                 persistent_workers=options['persistent_workers'],
                                 worker_init_fn=seed_worker,
                                 generator=g)
    else:
        raise

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create model with the chosen backbone
    model = UNet(input_bands=options['input_channels'],
                 output_classes=options['output_channels'],
                 hidden_channels=options['hidden_channels'],
                 backbone=options['backbone'])

    model.to(device)

    # Chargement du checkpoint
    if options['resume_from_epoch'] > 1:
        resume_model_dir = os.path.join(options['checkpoint_path'], str(options['resume_from_epoch']))
        model_file = os.path.join(resume_model_dir, options['checkpoint_name'])
        logging.info('Resuming training from epoch %d', options['resume_from_epoch'])
        logging.info('Loading model from: %s', model_file)

        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info('Model loaded successfully.')

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif options.get('pretrained_path'):
        logging.info('Loading pre-trained weights from: %s', options['pretrained_path'])
        checkpoint = torch.load(options['pretrained_path'], map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        logging.info('Pre-trained weights loaded (non-strict).')
    else:
        logging.info('Initializing model from scratch.')

    global class_distr, labels
    if options['agg_to_water']:
        agg_distr = sum(class_distr[-4:])
        class_distr[6] += agg_distr
        class_distr = class_distr[:-4]
        labels = labels[:-4]  # pour la matrice de confusion

    weight = gen_weights(class_distr, c=options['weight_param']).to(device)

    # Loss function
    if options['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=weight, gamma=options['focal_gamma'], ignore_index=-1, reduction='mean')
        logging.info("Using Focal Loss with gamma=%.2f", options['focal_gamma'])
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=weight)
        logging.info("Using CrossEntropy Loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['decay'])

    # Scheduler
    if options['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        logging.info("Using ReduceLROnPlateau scheduler")
    elif options['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['epochs'], eta_min=1e-6)
        logging.info("Using CosineAnnealing scheduler")
    else:  # multistep
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, options['lr_steps'], gamma=0.1)
        logging.info("Using MultiStepLR scheduler with steps %s", options['lr_steps'])

    start = options['resume_from_epoch'] + 1
    epochs = options['epochs']
    eval_every = options['eval_every']

    if options['mode'] == 'train':
        dataiter = iter(train_loader)
        image_temp, _ = next(dataiter)
        writer.add_graph(model, image_temp.to(device))

        model.train()

        for epoch in range(start, epochs + 1):
            training_loss = []
            training_batches = 0

            i_board = 0
            for (image, target) in tqdm(train_loader, desc="training"):
                image = image.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                logits = model(image)

                loss = criterion(logits, target)

                loss.backward()

                training_batches += target.shape[0]

                training_loss.append((loss.data * target.shape[0]).tolist())

                optimizer.step()

                writer.add_scalar('training loss', loss, (epoch - 1) * len(train_loader) + i_board)
                i_board += 1

            avg_train_loss = sum(training_loss) / training_batches
            logging.info("Epoch %d - Training loss: %.4f", epoch, avg_train_loss)

            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                test_loss = []
                test_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for (image, target) in tqdm(test_loader, desc="testing"):
                        image = image.to(device)
                        target = target.to(device)

                        logits = model(image)

                        loss = criterion(logits, target)

                        logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
                        logits = logits.reshape((-1, options['output_channels']))
                        target = target.reshape(-1)
                        mask = target != -1
                        logits = logits[mask]
                        target = target[mask]

                        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                        target = target.cpu().numpy()

                        test_batches += target.shape[0]
                        test_loss.append((loss.data * target.shape[0]).tolist())
                        y_predicted += probs.argmax(1).tolist()
                        y_true += target.tolist()

                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)

                average_val_loss = sum(test_loss) / test_batches
                acc = Evaluation(y_predicted, y_true)

                # Log consolidé train/val loss
                logging.info("Epoch %d - Train loss: %.4f - Val loss: %.4f", epoch, avg_train_loss, average_val_loss)

                # Matrice de confusion
                conf_mat = confusion_matrix(y_true, y_predicted, labels)
                logging.info("Confusion Matrix after epoch %d:\n%s", epoch, conf_mat.to_string())

                if average_val_loss < best_loss_value:
                    best_loss_value = average_val_loss
                    early_stop_counter = 0
                    logging.info("Best validation loss improved to: %.4f", best_loss_value)
                    logging.info("Evaluation after epoch %d: %s", epoch, acc)

                    model_dir = os.path.join(options['checkpoint_path'], str(epoch))
                    os.makedirs(model_dir, exist_ok=True)
                    best_model_path = os.path.join(model_dir, 'best_model.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logging.info("Best model saved to: %s", best_model_path)

                    writer.add_scalars('Loss per epoch', {'Test loss': average_val_loss,
                                                          'Train loss': avg_train_loss},
                                       epoch)
                else:
                    early_stop_counter += 1
                    logging.info("Validation loss did not improve. Early stop counter: %d/%d", early_stop_counter, patience)
                    if early_stop_counter >= patience:
                        logging.info('Early stopping triggered after epoch %d.', epoch)
                        print('Early stopping triggered.')
                        break

                # Écriture des métriques dans TensorBoard
                writer.add_scalar('Precision/test macroPrec', acc["macroPrec"], epoch)
                writer.add_scalar('Precision/test microPrec', acc["microPrec"], epoch)
                writer.add_scalar('Precision/test weightPrec', acc["weightPrec"], epoch)

                writer.add_scalar('Recall/test macroRec', acc["macroRec"], epoch)
                writer.add_scalar('Recall/test microRec', acc["microRec"], epoch)
                writer.add_scalar('Recall/test weightRec', acc["weightRec"], epoch)

                writer.add_scalar('F1/test macroF1', acc["macroF1"], epoch)
                writer.add_scalar('F1/test microF1', acc["microF1"], epoch)
                writer.add_scalar('F1/test weightF1', acc["weightF1"], epoch)

                writer.add_scalar('IoU/test MacroIoU', acc["IoU"], epoch)

                # Step scheduler
                if options['scheduler'] == 'plateau':
                    scheduler.step(average_val_loss)
                else:
                    scheduler.step()

                model.train()

    elif options['mode'] == 'test':
        model.eval()

        test_loss = []
        test_batches = 0
        y_true = []
        y_predicted = []

        with torch.no_grad():
            for (image, target) in tqdm(test_loader, desc="testing"):
                image = image.to(device)
                target = target.to(device)

                logits = model(image)

                loss = criterion(logits, target)

                logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
                logits = logits.reshape((-1, options['output_channels']))
                target = target.reshape(-1)
                mask = target != -1
                logits = logits[mask]
                target = target[mask]

                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                target = target.cpu().numpy()

                test_batches += target.shape[0]
                test_loss.append((loss.data * target.shape[0]).tolist())
                y_predicted += probs.argmax(1).tolist()
                y_true += target.tolist()

            y_predicted = np.asarray(y_predicted)
            y_true = np.asarray(y_true)

            acc = Evaluation(y_predicted, y_true)
            conf_mat = confusion_matrix(y_true, y_predicted, labels)
            logging.info("\n")
            logging.info("Test loss was: " + str(sum(test_loss) / test_batches))
            logging.info("STATISTICS: \n" + str(acc))
            logging.info("Confusion Matrix:\n" + conf_mat.to_string())
            print("Test loss:", sum(test_loss) / test_batches)
            print("Evaluation:", acc)
            print("Confusion Matrix:\n", conf_mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--agg_to_water', default=True, type=bool,
                        help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')

    parser.add_argument('--mode', default='train', help='select between train or test ')
    parser.add_argument('--epochs', default=150, type=int, help='Number of epochs to run')
    parser.add_argument('--batch', default=4, type=int, help='Batch size')
    parser.add_argument('--resume_from_epoch', default=0, type=int, help='load model from previous epoch')
    parser.add_argument('--pretrained_path', default=None, type=str, help='Path to pre-trained weights (optional)')
    parser.add_argument('--checkpoint_name', default='best_model.pth', type=str,
                        help='Name of the checkpoint file in the epoch folder (for resume)')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping')

    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    parser.add_argument('--hidden_channels', default=16, type=int, help='Number of hidden features')
    parser.add_argument('--backbone', default='none', type=str,
                        choices=['none', 'resnet18', 'mobilenetv2', 'efficientnetv2'],
                        help='Backbone encoder type (none for original UNet)')
    parser.add_argument('--weight_param', default=1.03, type=float, help='Weighting parameter for Loss Function')

    parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal'], help='Loss type')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Gamma for Focal Loss')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='learning rate decay')
    parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'multistep', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_steps', default='[40]', type=str, help='Steps for multistep scheduler')

    parser.add_argument('--checkpoint_path', default=os.path.join(up(os.path.abspath(__file__)), 'trained_models'),
                        help='folder to save checkpoints into')
    parser.add_argument('--eval_every', default=1, type=int, help='How frequently to run evaluation (epochs)')

    parser.add_argument('--num_workers', default=1, type=int,
                        help='How many cpus for loading data (0 is the main process)')
    parser.add_argument('--pin_memory', default=False, type=bool, help='Use pinned memory or not')
    parser.add_argument('--prefetch_factor', default=1, type=int,
                        help='Number of sample loaded in advance by each worker')
    parser.add_argument('--persistent_workers', default=True, type=bool,
                        help='This allows to maintain the workers Dataset instances alive.')
    parser.add_argument('--tensorboard', default='tsboard_segm', type=str, help='Name for tensorboard run')

    args = parser.parse_args()
    options = vars(args)

    # Parse lr_steps
    lr_steps = ast.literal_eval(options['lr_steps'])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise ValueError
    options['lr_steps'] = lr_steps

    logging.info('parsed input parameters:')
    logging.info(json.dumps(options, indent=2))
    main(options)