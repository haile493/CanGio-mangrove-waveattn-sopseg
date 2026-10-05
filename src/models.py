import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional

from src.config import Config


def build_segmentation_model(
    arch: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 9,
    classes: int = 1
) -> nn.Module:
    """
    Constructs a geospatial semantic segmentation model supporting arbitrary
    input spectral bands + indices.
    
    Supported architectures: 'unet', 'deeplabv3plus', 'fpn', 'pan'
    """
    arch = arch.lower()
    
    model_params = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": classes,
        "activation": None  # Return logits for stable loss computation
    }
    
    if arch == "unet":
        model = smp.Unet(**model_params)
    elif arch in ("deeplabv3+", "deeplabv3plus"):
        model = smp.DeepLabV3Plus(**model_params)
    elif arch == "segformer":
        model = smp.Segformer(**model_params)
    elif arch == "upernet":
        model = smp.UPerNet(**model_params)
    elif arch == "fpn":
        model = smp.FPN(**model_params)
    elif arch == "pan":
        model = smp.PAN(**model_params)
    elif arch == "unetplusplus":
        model = smp.UnetPlusPlus(**model_params)
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Choose from unet, deeplabv3plus, segformer, upernet, fpn, pan, unetplusplus")
        
    return model


STUDY3_MODEL_CONFIGS = {
    "unet_resnet50": {
        "arch": "unet",
        "encoder_name": "resnet50",
        "display_name": "U-Net (ResNet-50)"
    },
    "deeplabv3plus_effnetb4": {
        "arch": "deeplabv3plus",
        "encoder_name": "timm-efficientnet-b4",
        "display_name": "DeepLabV3+ (EffNet-B4)"
    },
    "segformer_mitb5": {
        "arch": "segformer",
        "encoder_name": "mit_b5",
        "display_name": "SegFormer (MiT-B5)"
    },
    "upernet_mitb5": {
        "arch": "upernet",
        "encoder_name": "mit_b5",
        "display_name": "UPerNet (MiT-B5)"
    }
}


class AutoPadModelWrapper(nn.Module):
    """
    Wraps a model to dynamically pad spatial dimensions to multiples of 32 (or 16),
    satisfying downsampling constraints of deep encoders (EffNet-B4, MiT-B5, ASPP, UPerNet),
    and unpads the resulting logits back to the exact original input dimensions.
    """
    def __init__(self, base_model: nn.Module, multiple: int = 32):
        super().__init__()
        self.base_model = base_model
        self.multiple = multiple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple

        if pad_h > 0 or pad_w > 0:
            x_pad = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            out = self.base_model(x_pad)
            return out[:, :, :h, :w]
        return self.base_model(x)


def build_study3_model(model_key: str, in_channels: int = 9, encoder_weights: Optional[str] = "imagenet") -> nn.Module:
    """
    Constructs one of the 4 models defined in Dinh et al. (2026):
    - 'unet_resnet50'
    - 'deeplabv3plus_effnetb4'
    - 'segformer_mitb5'
    - 'upernet_mitb5'
    """
    if model_key not in STUDY3_MODEL_CONFIGS:
        raise KeyError(f"Unknown Study 3 model key: {model_key}. Available: {list(STUDY3_MODEL_CONFIGS.keys())}")
    
    cfg = STUDY3_MODEL_CONFIGS[model_key]
    raw_model = build_segmentation_model(
        arch=cfg["arch"],
        encoder_name=cfg["encoder_name"],
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1
    )
    return AutoPadModelWrapper(raw_model, multiple=32)



def get_model_from_config(config: Config) -> nn.Module:
    """Instantiates model directly from a Config instance."""
    return build_segmentation_model(
        arch=config.arch,
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.num_classes
    )


# Re-export temporal architectures for convenient access
from src.temporal_models import (
    SiameseUNet,
    EarlyFusionBiTemporalUNet,
    TemporalConvLSTM_UNet,
    TemporalAttentionUNet,
    build_bitemporal_model,
    build_sequential_model
)


