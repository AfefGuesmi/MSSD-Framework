# coding=utf-8
"""Swin‑UNet wrapper with pre‑training adaptation."""

from __future__ import absolute_import, division, print_function

import logging
import math

import torch
import torch.nn as nn

from swin_unet_v2 import SwinTransformerSys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class SwinUnet(nn.Module):
    """
    Swin‑UNet for semantic segmentation.

    Args:
        config (Config): Configuration object.
        img_size (int): Input image size.
        num_classes (int): Number of output classes.
        zero_head (bool): If True, initialise output layer with zeros.
        vis (bool): Unused, kept for compatibility.
    """

    def __init__(self, config, img_size=224, num_classes=11, zero_head=False, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.SWIN.DROP_RATE,
            drop_path_rate=config.MODEL.SWIN.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

        logger.info("Initialized SwinUnet with configuration: %s", config)

    def forward(self, x):
        """Forward pass. If input has 1 channel, replicate to 3 channels."""
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_pretrained(self, pretrained_path):
        """
        Load pre‑trained Swin‑v1 weights and adapt the first convolution.

        The original patch_embed.proj expects 3 input channels. We adapt it to
        11 channels by copying the first 3 and initialising the rest with
        Kaiming normal.

        Args:
            pretrained_path (str): Path to the checkpoint file.
        """
        logger.info("Loading pre-trained weights from: %s", pretrained_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained_path, map_location=device)

        state_dict = checkpoint.get('model', checkpoint)
        model_dict = self.swin_unet.state_dict()

        # Special handling for patch_embed.proj.weight
        if 'patch_embed.proj.weight' in state_dict:
            pretrained_w = state_dict['patch_embed.proj.weight']  # (out_c, 3, kH, kW)
            out_c, _, kH, kW = pretrained_w.shape
            target_shape = model_dict['patch_embed.proj.weight'].shape  # (out_c, 11, kH, kW)
            new_weight = torch.zeros(target_shape, device=device)

            # Copy the first 3 channels
            new_weight[:, :3, :, :] = pretrained_w

            # Kaiming normal initialisation for the remaining 8 channels
            fan_out = out_c * kH * kW
            std = math.sqrt(2.0 / fan_out)
            nn.init.normal_(new_weight[:, 3:, :, :], mean=0.0, std=std)

            state_dict['patch_embed.proj.weight'] = new_weight
            logger.info(
                "Adapted patch_embed.proj.weight: copied first 3 channels, "
                "initialized remaining 8 with Kaiming normal (std=%.4f)", std
            )

        # Filter out keys with shape mismatches (e.g., decoder, output layer)
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    logger.warning(
                        "Shape mismatch for %s: pretrained %s, model %s. Skipping.",
                        k, v.shape, model_dict[k].shape
                    )
            else:
                logger.debug("Key %s not found in model, skipping.", k)

        msg = self.swin_unet.load_state_dict(filtered_dict, strict=False)
        logger.info("Loaded pre-trained weights (non-strict) with message: %s", msg)

    def load_from(self, config):
        """
        Load pre‑trained weights if specified in config.

        Args:
            config (Config): Configuration object.
        """
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        else:
            logger.warning("No pre-trained checkpoint found, initializing model from scratch.")