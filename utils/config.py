"""Configuration for Swin‑UNet model."""


class Config:
    """Central configuration class."""

    class DATA:
        """Data-related parameters."""
        IMG_SIZE = 256  # Input patch size

    class MODEL:
        """Model architecture parameters."""

        SWIN = type('Swin', (), {
            'PATCH_SIZE': 4,
            'IN_CHANS': 11,
            'EMBED_DIM': 96,
            'DEPTHS': [2, 2, 2, 2],
            'NUM_HEADS': [3, 6, 12, 24],
            'WINDOW_SIZE': 8,          # compatible with 256//4=64, 64 divisible by 8
            'MLP_RATIO': 4.0,
            'QKV_BIAS': True,
            'QK_SCALE': None,
            'DROP_RATE': 0.0,
            'DROP_PATH_RATE': 0.1,
            'APE': False,
            'PATCH_NORM': True
        })

    class TRAIN:
        """Training-related parameters."""
        USE_CHECKPOINT = False