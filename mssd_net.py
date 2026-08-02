# -*- coding: utf-8 -*-
"""
mssd_net.py

MSSD-Net: an enhanced Swin-UNet V2 for marine surface debris segmentation.

This module adds three real architectural contributions on top of the
unmodified swin_unet_v2.SwinTransformerSys building blocks, closing the gap
between what the MSSD paper claims and what the code actually does:

  1. DilatedBottleneck  -- an ASPP-style multi-scale dilated-convolution
     module applied to the encoder bottleneck, wrapped in a residual
     connection. Captures multi-scale spatial context (useful for thin,
     elongated debris trails) that pure 7x7 window attention can miss.

  2. AttentionRefinement -- a CBAM-style channel + spatial attention gate
     applied in the decoder, right after encoder/decoder features are
     fused. Suppresses background clutter (sea foam, sun glint, turbid
     water) relative to debris-relevant regions before each decoder stage
     refines the feature map further.

  3. Residual skip fusion -- the encoder/decoder fusion at each decoder
     stage is now F_dec = F_up + Linear(Concat(F_enc, F_up)) instead of
     F_dec = Linear(Concat(F_enc, F_up)), giving the fused feature an
     identity shortcut around the fusion projection and easing
     optimisation of the deep decoder stack.

All other components (patch embedding, Swin Transformer V2 encoder blocks
with scaled cosine attention, patch merging/expanding, the final 4x
patch-expand + 1x1 conv head) are reused unchanged from swin_unet_v2.py.

Usage:
    from mssd_net import MSSDNet
    model = MSSDNet(img_size=256, patch_size=4, in_chans=11, num_classes=11,
                     embed_dim=96, depths=[2,2,2,2], depths_decoder=[1,2,2,2],
                     num_heads=[3,6,12,24], window_size=8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from swin_unet_v2 import (
    PatchEmbed, BasicLayer, BasicLayer_up, PatchMerging, PatchExpand,
    FinalPatchExpand_X4,
)


class DilatedBottleneck(nn.Module):
    """
    ASPP-lite multi-scale dilated-convolution module for the encoder
    bottleneck, with a residual connection around the whole block.

    Operates on a 2D feature map (B, C, H, W). Parallel branches with
    increasing dilation rates enlarge the receptive field without losing
    spatial resolution, and a global-pooling branch adds image-level
    context. Branch outputs are concatenated and projected back to the
    input channel width, then added back to the input (residual).
    """

    def __init__(self, channels, dilations=(1, 2, 4)):
        super().__init__()
        branch_channels = max(channels // 4, 16)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        fused_channels = branch_channels * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(fused_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]

        global_feat = self.global_branch(x)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        feats.append(global_feat)

        fused = self.project(torch.cat(feats, dim=1))
        return self.act(x + fused)  # residual connection (Eq. y = F(x) + x)


class AttentionRefinement(nn.Module):
    """
    CBAM-lite decoder attention refinement: sequential channel attention
    then spatial attention, applied to a 2D feature map (B, C, H, W).

    Channel attention re-weights *which* feature channels matter (e.g.
    spectral-response channels that best separate debris from water);
    spatial attention re-weights *where* in the image matters (e.g.
    suppressing large uniform background regions). Matches Eq. F' = A*F,
    A = sigma(W*F+b) from the paper, implemented as two lightweight gates
    rather than a single conv.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ---- channel attention ----
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        channel_att = self.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel_att.unsqueeze(-1).unsqueeze(-1)

        # ---- spatial attention ----
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        x = x * spatial_att

        return x


def _to_2d(x, H, W):
    """(B, L, C) -> (B, C, H, W)"""
    B, L, C = x.shape
    return x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def _to_seq(x):
    """(B, C, H, W) -> (B, L, C)"""
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)


class MSSDNet(nn.Module):
    """
    MSSD-Net: Swin-UNet V2 with a residual dilated bottleneck and
    residual, attention-refined decoder skip fusion.

    Constructor signature matches swin_unet_v2.SwinTransformerSys so it
    can be dropped in as a replacement with no other code changes.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=11, num_classes=11,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True, use_checkpoint=False,
                 final_upsample="expand_first",
                 use_dilated_bottleneck=True,
                 use_attention_refinement=True,
                 use_residual_fusion=True,
                 **kwargs):
        super().__init__()

        # ---- Ablation switches: each of the three contributions can be
        # toggled independently, so the *same* class produces every point
        # in the ablation table (baseline, each addition in isolation, any
        # partial combination, and the full MSSD-Net). ----
        self.use_dilated_bottleneck = use_dilated_bottleneck
        self.use_attention_refinement = use_attention_refinement
        self.use_residual_fusion = use_residual_fusion

        variant_bits = [
            ('dilated-bottleneck', use_dilated_bottleneck),
            ('attention-refinement', use_attention_refinement),
            ('residual-fusion', use_residual_fusion),
        ]
        active = [name for name, on in variant_bits if on] or ['none (plain Swin-UNet V2)']
        print("MSSDNet ---- active contributions: {}".format(', '.join(active)))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # ---- Patch embedding (unchanged) ----
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ---- Encoder + bottleneck (unchanged) ----
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                   patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # ---- Bottleneck: residual dilated multi-scale context (ASPP-lite) ----
        self.bottleneck_resolution = (
            patches_resolution[0] // (2 ** (self.num_layers - 1)),
            patches_resolution[1] // (2 ** (self.num_layers - 1)),
        )
        if self.use_dilated_bottleneck:
            self.dilated_bottleneck = DilatedBottleneck(self.num_features, dilations=(1, 2, 4))
        else:
            self.dilated_bottleneck = None

        # ---- Decoder (structure unchanged, fusion logic modified in forward) ----
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.decoder_refine = nn.ModuleList()  # NEW: attention refinement per decoder stage

        for i_layer in range(self.num_layers):
            stage_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            concat_linear = nn.Linear(2 * stage_dim, stage_dim) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                       patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=stage_dim, dim_scale=2, norm_layer=norm_layer)
                self.decoder_refine.append(nn.Identity())  # no fusion at stage 0 (bottleneck -> first up)
            else:
                layer_up = BasicLayer_up(
                    dim=stage_dim,
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                       patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):
                                  sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)
                if self.use_attention_refinement:
                    self.decoder_refine.append(AttentionRefinement(stage_dim, reduction=8))
                else:
                    self.decoder_refine.append(nn.Identity())

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes,
                                    kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # ---------------- Encoder + bottleneck ----------------
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B, L, C  (C = num_features, deepest resolution)

        # ---- Bottleneck: residual dilated multi-scale context (optional) ----
        if self.use_dilated_bottleneck:
            Hb, Wb = self.bottleneck_resolution
            x_2d = _to_2d(x, Hb, Wb)
            x_2d = self.dilated_bottleneck(x_2d)
            x = _to_seq(x_2d)

        return x, x_downsample

    # ---------------- Decoder + skip fusion (residual + attention refinement optional) ----------------
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                skip = x_downsample[3 - inx]
                x_up = x  # upsampled decoder feature, dim = stage_dim

                fused = torch.cat([x_up, skip], -1)              # dim = 2 * stage_dim
                fused = self.concat_back_dim[inx](fused)           # dim = stage_dim

                if self.use_residual_fusion:
                    x = x_up + fused                               # residual skip fusion
                else:
                    x = fused                                      # plain fusion (original Swin-UNet)

                if self.use_attention_refinement:
                    res = self.layers_up[inx].input_resolution
                    x_2d = _to_2d(x, res[0], res[1])
                    x_2d = self.decoder_refine[inx](x_2d)
                    x = _to_seq(x_2d)

                x = layer_up(x)

        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x

    # ---------------- ImageNet Swin V2 encoder initialisation ----------------
    def load_pretrained(self, pretrained_path, verbose=True):
        """
        Initialise the encoder (patch embedding + the four Swin V2 stages
        in self.layers) from an ImageNet-pretrained Swin V2 checkpoint,
        e.g. swinv2_tiny_patch4_window8_256.pth from the official
        microsoft/Swin-Transformer release.

        Only encoder-side tensors are ever looked up in the checkpoint:
        patch_embed.*, layers.*, and the post-encoder norm.*. Everything
        else -- the decoder (layers_up, concat_back_dim, decoder_refine),
        the classification head, and the MSSD-specific dilated_bottleneck
        -- has no ImageNet counterpart and is left at its random init.

        Two things transfer imperfectly rather than not at all, worth
        knowing about:
          * This encoder's transformer blocks are pre-norm (norm1/norm2
            applied before attn/mlp); the released checkpoint was trained
            post-norm. Names and shapes still line up so the weights load,
            but their role in the forward pass isn't identical.
          * The relative-position-bias path here (self.cpb + self.tau) is
            a custom design, not the official cpb_mlp/logit_scale, so it
            has no matching name in the checkpoint and stays randomly
            initialised.

        Args:
            pretrained_path (str): path to the .pth checkpoint.
            verbose (bool): print a one-line summary if True.

        Returns:
            dict with four lists of parameter names:
              'loaded'         -- copied as-is (name + shape matched)
              'adapted'        -- copied with a shape adjustment (only the
                                   patch-embedding conv, for in_chans != 3)
              'skipped'        -- name matched but shape didn't, left at
                                   random init
              'not_applicable' -- no corresponding name in the checkpoint
                                   at all (decoder / MSSD-specific tensors)
        """
        ckpt = torch.load(pretrained_path, map_location='cpu')
        if isinstance(ckpt, dict):
            for wrapper_key in ('model', 'state_dict'):
                if wrapper_key in ckpt and isinstance(ckpt[wrapper_key], dict):
                    ckpt = ckpt[wrapper_key]
                    break
        pretrained_sd = ckpt

        # Only these are ever considered "encoder" keys; everything else
        # (layers_up, concat_back_dim, decoder_refine, dilated_bottleneck,
        # norm_up, up, output, ...) is intentionally skipped without even
        # checking the checkpoint for a same-named tensor.
        encoder_prefixes = ('patch_embed.', 'layers.')
        encoder_exact = ('norm.weight', 'norm.bias')

        own_sd = self.state_dict()
        new_sd = {}
        loaded, adapted, skipped, not_applicable = [], [], [], []

        for key, own_tensor in own_sd.items():
            is_encoder_key = key.startswith(encoder_prefixes) or key in encoder_exact
            if not is_encoder_key:
                not_applicable.append(key)
                continue

            if key not in pretrained_sd:
                not_applicable.append(key)
                continue

            pre_tensor = pretrained_sd[key]

            if pre_tensor.shape == own_tensor.shape:
                new_sd[key] = pre_tensor.clone()
                loaded.append(key)
            elif (key == 'patch_embed.proj.weight'
                  and pre_tensor.shape[0] == own_tensor.shape[0]
                  and pre_tensor.shape[2:] == own_tensor.shape[2:]):
                new_sd[key] = self._adapt_patch_embed_conv(pre_tensor, own_tensor.shape)
                adapted.append(key)
            else:
                # Same name exists in the checkpoint but the shape doesn't
                # match (e.g. this encoder's PatchMerging norm is
                # pre-reduction/4*dim while the official one is
                # post-reduction/2*dim). Leave at random init.
                skipped.append(key)

        own_sd.update(new_sd)
        self.load_state_dict(own_sd, strict=True)

        if verbose:
            print('MSSDNet.load_pretrained: {} loaded, {} channel-adapted, '
                  '{} skipped (name matched, shape mismatch), {} left at '
                  'random init (no ImageNet counterpart).'.format(
                      len(loaded), len(adapted), len(skipped), len(not_applicable)))

        return {'loaded': loaded, 'adapted': adapted, 'skipped': skipped,
                'not_applicable': not_applicable}

    @staticmethod
    def _adapt_patch_embed_conv(pretrained_weight, target_shape):
        """
        Adapt an ImageNet patch-embedding conv kernel, shaped
        (embed_dim, 3, p, p), to a different input channel count, shaped
        (embed_dim, in_chans, p, p).

        Averages the 3 RGB input-channel filters into a single
        channel-agnostic filter, then repeats it across the target number
        of input channels -- the standard trick for reusing RGB ImageNet
        weights on multispectral input (here, 11 Sentinel-2 bands).
        """
        out_c, in_c, kh, kw = target_shape
        if pretrained_weight.shape[1] != 3:
            # Unexpected pretrained format; don't guess, just random-init.
            return torch.empty(target_shape).normal_(std=.02)
        mean_kernel = pretrained_weight.mean(dim=1, keepdim=True)  # (embed_dim, 1, p, p)
        return mean_kernel.repeat(1, in_c, 1, 1).contiguous()


# ---------------------------------------------------------------------------
# Ablation study support
# ---------------------------------------------------------------------------
# Each of the three contributions is an independent boolean flag, so any of
# the 2^3 = 8 combinations can be built. The dict below covers the two
# ablation stories that matter most in practice:
#
#   * Cumulative / progressive (the typical paper ablation table): start
#     from the plain Swin-UNet V2 baseline and add one contribution at a
#     time until reaching the full MSSD-Net.
#   * Isolated (marginal contribution of each piece on its own, useful for
#     understanding which single addition matters most).
#
# Use build_mssd_net(variant, **kwargs) to construct any of them, or pass
# the three use_* flags directly to MSSDNet(...) for any other combination.
VARIANTS = {
    # ---- cumulative / progressive ablation ----
    'baseline':                 dict(use_dilated_bottleneck=False, use_attention_refinement=False, use_residual_fusion=False),
    '+dilated':                 dict(use_dilated_bottleneck=True,  use_attention_refinement=False, use_residual_fusion=False),
    '+dilated+attention':       dict(use_dilated_bottleneck=True,  use_attention_refinement=True,  use_residual_fusion=False),
    'full':                     dict(use_dilated_bottleneck=True,  use_attention_refinement=True,  use_residual_fusion=True),

    # ---- isolated / marginal-contribution ablation ----
    'only_dilated':             dict(use_dilated_bottleneck=True,  use_attention_refinement=False, use_residual_fusion=False),
    'only_attention':           dict(use_dilated_bottleneck=False, use_attention_refinement=True,  use_residual_fusion=False),
    'only_residual':            dict(use_dilated_bottleneck=False, use_attention_refinement=False, use_residual_fusion=True),
}


def build_mssd_net(variant='full', **kwargs):
    """
    Build an MSSDNet for a named ablation variant.

    Args:
        variant (str): one of VARIANTS.keys(), e.g. 'baseline', '+dilated',
            '+dilated+attention', 'full', 'only_dilated', 'only_attention',
            'only_residual'.
        **kwargs: any other MSSDNet constructor argument (img_size,
            embed_dim, window_size, num_classes, ...).

    Returns:
        MSSDNet instance configured for the requested variant.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANTS.keys())}")
    flags = VARIANTS[variant]
    return MSSDNet(**flags, **kwargs)