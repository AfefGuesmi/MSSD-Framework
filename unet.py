# -*- coding: utf-8 -*-
'''
Original Paper: Ronneberger et al., 2015 (https://arxiv.org/abs/1505.04597)
Initial Pytorch Implementation: Alexandre Milesi (https://github.com/milesial/Pytorch-UNet)
This modified implementation: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Python Version: 3.7.10
Description: unet.py Unet model for pixel-level semantic segmentation with pluggable backbones.
'''

import torch
import numpy as np
from torch import nn
import random
import torchvision.models as models
from collections import OrderedDict

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class Down(nn.Module):
    """Contracting Layer (used only when backbone='none')"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Expanding Layer (used for all decoder blocks)"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle possible size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ----------------------------------------------------------------------
# Backbone encoders
# ----------------------------------------------------------------------
class ResNetEncoder(nn.Module):
    """ResNet18 encoder (returns list of skip features from high res to deep)"""
    def __init__(self, in_channels=3):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print(f"Warning: Input channels changed to {in_channels}, first conv layer randomly initialised.")

        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu),
            ('maxpool', resnet.maxpool),
            ('layer1', resnet.layer1),
            ('layer2', resnet.layer2),
            ('layer3', resnet.layer3),
            ('layer4', resnet.layer4),
        ]))

    def forward(self, x):
        features = []
        for name, layer in self.encoder.named_children():
            x = layer(x)
            if name in ['maxpool', 'layer1', 'layer2', 'layer3', 'layer4']:
                features.append(x)
        return features  # from high res to low res (deepest last)


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 encoder"""
    def __init__(self, in_channels=3):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            mobilenet.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            print(f"Warning: Input channels changed to {in_channels}, first conv layer randomly initialised.")

        self.features = mobilenet.features
        self.out_indices = [1, 2, 4, 7, 14]   # typical for MobileNetV2

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                features.append(x)
        return features


class EfficientNetV2Encoder(nn.Module):
    """EfficientNetV2‑S encoder"""
    def __init__(self, in_channels=3):
        super().__init__()
        effnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            effnet.features[0][0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
            print(f"Warning: Input channels changed to {in_channels}, first conv layer randomly initialised.")

        self.features = effnet.features
        self.out_indices = [1, 2, 3, 5, 7]   # typical for EfficientNetV2‑S

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                features.append(x)
        return features


# ----------------------------------------------------------------------
# Main UNet class with dynamic decoder construction
# ----------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, input_bands=11, output_classes=11, hidden_channels=16, backbone='none'):
        super(UNet, self).__init__()
        self.backbone = backbone
        self.input_bands = input_bands

        if backbone == 'none':
            # Original UNet
            self.inc = nn.Sequential(
                nn.Conv2d(input_bands, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
            self.down1 = Down(hidden_channels, 2 * hidden_channels)
            self.down2 = Down(2 * hidden_channels, 4 * hidden_channels)
            self.down3 = Down(4 * hidden_channels, 8 * hidden_channels)
            self.down4 = Down(8 * hidden_channels, 8 * hidden_channels)

            self.up1 = Up(16 * hidden_channels, 4 * hidden_channels)
            self.up2 = Up(8 * hidden_channels, 2 * hidden_channels)
            self.up3 = Up(4 * hidden_channels, hidden_channels)
            self.up4 = Up(2 * hidden_channels, hidden_channels)

            self.outc = nn.Conv2d(hidden_channels, output_classes, kernel_size=1)
        else:
            # Select encoder
            if backbone == 'resnet18':
                self.encoder = ResNetEncoder(in_channels=input_bands)
            elif backbone == 'mobilenetv2':
                self.encoder = MobileNetV2Encoder(in_channels=input_bands)
            elif backbone == 'efficientnetv2':
                self.encoder = EfficientNetV2Encoder(in_channels=input_bands)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            self._build_decoder_from_encoder(output_classes)

    def _build_decoder_from_encoder(self, output_classes):
        """Build decoder dynamically using dummy forward to get actual skip shapes and resolutions."""
        device = torch.device('cpu')
        dummy = torch.zeros(1, self.input_bands, 256, 256, device=device)
        with torch.no_grad():
            skip_features = self.encoder(dummy)   # list from high res to low res

        skip_channels = [f.shape[1] for f in skip_features]
        skip_sizes = [f.shape[2] for f in skip_features]
        print(f"[{self.backbone}] Skip channels: {skip_channels}, sizes: {skip_sizes}")

        # We need to restore to input size (256). The highest skip size is the first element.
        target_size = dummy.shape[2]
        current_size = skip_sizes[0]
        extra_ups = 0
        while current_size < target_size:
            current_size *= 2
            extra_ups += 1

        # Reverse skip list (deepest first) for decoder
        rev_channels = list(reversed(skip_channels))
        rev_sizes = list(reversed(skip_sizes))

        # Build the standard decoder blocks (one per skip, excluding the last deepest)
        num_blocks = len(rev_channels) - 1
        dec_out_channels = []
        base = max(16, rev_channels[-1] // 8)
        for i in range(num_blocks):
            if i == 0:
                out_ch = rev_channels[1]
            elif i == 1:
                out_ch = max(rev_channels[2] // 2, base)
            elif i == 2:
                out_ch = max(rev_channels[3] // 2, base)
            else:
                out_ch = base
            dec_out_channels.append(out_ch)

        self.up_blocks = nn.ModuleList()
        in_ch = rev_channels[0]   # deepest
        for i in range(num_blocks):
            skip_ch = rev_channels[i + 1]
            concat_ch = in_ch + skip_ch
            out_ch = dec_out_channels[i]
            self.up_blocks.append(Up(concat_ch, out_ch))
            in_ch = out_ch

        # Now add extra up blocks to reach full resolution
        self.extra_up_blocks = nn.ModuleList()
        for _ in range(extra_ups):
            # Each extra up block: input channels = in_ch, output channels = in_ch // 2
            # No skip connection from encoder (we could use input image, but for simplicity just upsample)
            # We'll use a simple Upsample + Conv
            extra_up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_ch // 2),
                nn.ReLU(inplace=True)
            )
            self.extra_up_blocks.append(extra_up)
            in_ch = in_ch // 2

        # Final convolution
        self.outc = nn.Conv2d(in_ch, output_classes, kernel_size=1)

    def forward(self, x):
        if self.backbone == 'none':
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x6 = self.up1(x5, x4)
            x7 = self.up2(x6, x3)
            x8 = self.up3(x7, x2)
            x9 = self.up4(x8, x1)

            logits = self.outc(x9)
            return logits
        else:
            features = self.encoder(x)          # high res to low res
            features = list(reversed(features)) # deepest first
            x = features[0]
            for i, up in enumerate(self.up_blocks):
                x = up(x, features[i+1])
            for extra_up in self.extra_up_blocks:
                x = extra_up(x)
            logits = self.outc(x)
            return logits