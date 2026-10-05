import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class AutoPadWrapper(nn.Module):
    """
    Dynamically pads spatial (H, W) dimensions to a multiple of `multiple` (default 32)
    to prevent dimension mismatches in multi-scale encoders/decoders, then unpads back.
    Supports 4D tensors (B, C, H, W), 5D sequence tensors (B, T, C, H, W), or bi-temporal inputs.
    """
    def __init__(self, base_model: nn.Module, multiple: int = 32):
        super().__init__()
        self.base_model = base_model
        self.multiple = multiple

    def _pad_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple
        if pad_h == 0 and pad_w == 0:
            return x, h, w

        if x.dim() == 4:
            # (B, C, H, W)
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        elif x.dim() == 5:
            # (B, T, C, H, W)
            b, t, c, _, _ = x.shape
            x_flat = x.view(b * t, c, h, w)
            x_pad = F.pad(x_flat, (0, pad_w, 0, pad_h), mode="reflect").view(b, t, c, h + pad_h, w + pad_w)
        else:
            x_pad = x
        return x_pad, h, w

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 1:
            x = args[0]
            x_pad, orig_h, orig_w = self._pad_tensor(x)
            out = self.base_model(x_pad, **kwargs)
            return out[..., :orig_h, :orig_w]
        elif len(args) == 2:
            x1, x2 = args
            x1_pad, orig_h, orig_w = self._pad_tensor(x1)
            x2_pad, _, _ = self._pad_tensor(x2)
            out = self.base_model(x1_pad, x2_pad, **kwargs)
            return out[..., :orig_h, :orig_w]
        else:
            return self.base_model(*args, **kwargs)


# =========================================================================
# OPTION 2: Siamese 2D ResNet-34 / ResNet-50 U-Net for Bi-Temporal Imagery
# =========================================================================

class SiameseFeatureFusionBlock(nn.Module):
    """
    Multi-scale bi-temporal fusion:
    Takes features F1 (pre) and F2 (post) at scale i, computes:
    1. Absolute spectral difference: |F1 - F2|
    2. Concat([|F1 - F2|, F2]) passed through 1x1 conv + BN + ReLU
    This captures both change magnitude and post-disturbance land cover state.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        fused = torch.cat([diff, f2], dim=1)
        return self.conv(fused)


class SiameseUNet(nn.Module):
    """
    Option 2: Siamese 2D U-Net with ResNet-34 or ResNet-50 Backbone.
    Extracts multi-scale representations of pre- and post-disturbance imagery
    via a weight-shared 2D ResNet encoder, computes multi-scale feature differences,
    and decodes them using a U-Net decoder to predict disturbance zones.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 9,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels

        self.fusions = nn.ModuleList([
            SiameseFeatureFusionBlock(ch) for ch in enc_channels
        ])

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            n_blocks=5
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: (B, C, H, W) pre-disturbance image
        x2: (B, C, H, W) post-disturbance image
        """
        feats1 = self.encoder(x1)
        feats2 = self.encoder(x2)

        fused_feats = [
            fusion(f1, f2) for f1, f2, fusion in zip(feats1, feats2, self.fusions)
        ]

        decoder_output = self.decoder(fused_feats)
        return self.segmentation_head(decoder_output)


class EarlyFusionBiTemporalUNet(nn.Module):
    """
    Option 2 (Variant): Early-Fusion Bi-Temporal U-Net.
    Concatenates pre (X1) and post (X2) imagery along spectral channels
    (e.g., 9 + 9 = 18 channels) and feeds them directly into a 2D ResNet U-Net.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels_per_image: int = 9,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels_per_image * 2,
            classes=classes
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x_concat = torch.cat([x1, x2], dim=1)
        return self.model(x_concat)


# =========================================================================
# OPTION 3: 2D ResNet-34 / ResNet-50 + ConvLSTM / Temporal Attention
# =========================================================================

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell for 2D spatial feature dynamics over time.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            batch_size, _, h, w = x.shape
            h_cur = torch.zeros(batch_size, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
            c_cur = torch.zeros(batch_size, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
        else:
            h_cur, c_cur = state

        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)


class TemporalConvLSTM_UNet(nn.Module):
    """
    Option 3A: 2D ResNet Encoder + Multi-Scale ConvLSTM + U-Net Decoder.
    Takes a temporal sequence of satellite images (B, T, C, H, W).
    Extracts spatial representations with shared 2D ResNet (ResNet-34 / ResNet-50)
    for each time step, recursively aggregates spatiotemporal change via ConvLSTM cells,
    and reconstructs the disturbance map via U-Net decoder.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 9,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels

        # ConvLSTM cells across each feature scale
        self.temporal_cells = nn.ModuleList([
            ConvLSTMCell(ch, ch, kernel_size=3) for ch in enc_channels
        ])

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            n_blocks=5
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, T, C, H, W)
        """
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        all_feats = self.encoder(x_flat)

        fused_feats = []
        for feat, cell in zip(all_feats, self.temporal_cells):
            _, ch, hi, wi = feat.shape
            feat_seq = feat.view(B, T, ch, hi, wi)
            state = None
            for t in range(T):
                h_t, state = cell(feat_seq[:, t], state)
            fused_feats.append(h_t)

        decoder_out = self.decoder(fused_feats)
        return self.segmentation_head(decoder_out)


class TemporalSelfAttentionModule(nn.Module):
    """
    Option 3B module: Pixel-wise Temporal Attention (Transformer / U-TAE style).
    Attends across timestamps T to learn dynamic weights for disturbance detection.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, max(16, channels // 4), kernel_size=1)
        self.k = nn.Conv2d(channels, max(16, channels // 4), kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_seq: torch.Tensor) -> torch.Tensor:
        """
        feat_seq: (B, T, C, H, W)
        """
        B, T, C, H, W = feat_seq.shape
        x_flat = feat_seq.view(B * T, C, H, W)
        q = self.q(x_flat).view(B, T, -1, H * W).permute(0, 3, 1, 2)  # (B, HW, T, C_k)
        k = self.k(x_flat).view(B, T, -1, H * W).permute(0, 3, 2, 1)  # (B, HW, C_k, T)
        v = self.v(x_flat).view(B, T, C, H * W).permute(0, 3, 1, 2)   # (B, HW, T, C)

        scale = 1.0 / math.sqrt(q.shape[-1])
        attn = torch.softmax(torch.matmul(q, k) * scale, dim=-1)      # (B, HW, T, T)
        attended = torch.matmul(attn, v)                              # (B, HW, T, C)
        
        # Aggregate across time (average or target final timestamp)
        aggregated = attended.mean(dim=2).permute(0, 2, 1).view(B, C, H, W)
        return feat_seq[:, -1] + self.gamma * aggregated


class TemporalAttentionUNet(nn.Module):
    """
    Option 3B: 2D ResNet Encoder + Temporal Attention + U-Net Decoder.
    Captures temporal dependencies across satellite passes via multi-scale self-attention.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 9,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels

        self.temporal_attns = nn.ModuleList([
            TemporalSelfAttentionModule(ch) for ch in enc_channels
        ])

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            n_blocks=5
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, T, C, H, W)
        """
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        all_feats = self.encoder(x_flat)

        fused_feats = []
        for feat, attn in zip(all_feats, self.temporal_attns):
            _, ch, hi, wi = feat.shape
            feat_seq = feat.view(B, T, ch, hi, wi)
            fused_feats.append(attn(feat_seq))

        decoder_out = self.decoder(fused_feats)
        return self.segmentation_head(decoder_out)


# =========================================================================
# FACTORY BUILDERS
# =========================================================================

def build_bitemporal_model(
    model_type: str = "siamese",
    encoder_name: str = "resnet34",
    in_channels: int = 9,
    encoder_weights: Optional[str] = "imagenet"
) -> nn.Module:
    """
    Factory for Option 2 models:
    - 'siamese': SiameseUNet with shared ResNet backbone & multi-scale feature differencing.
    - 'early_fusion': EarlyFusionBiTemporalUNet with 2x channel concatenated input.
    """
    if model_type == "siamese":
        base_model = SiameseUNet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            encoder_weights=encoder_weights
        )
    elif model_type in ("early_fusion", "concat"):
        base_model = EarlyFusionBiTemporalUNet(
            encoder_name=encoder_name,
            in_channels_per_image=in_channels,
            encoder_weights=encoder_weights
        )
    else:
        raise ValueError(f"Unknown bi-temporal model_type: {model_type}")

    return AutoPadWrapper(base_model, multiple=32)


def build_sequential_model(
    model_type: str = "convlstm",
    encoder_name: str = "resnet34",
    in_channels: int = 9,
    encoder_weights: Optional[str] = "imagenet"
) -> nn.Module:
    """
    Factory for Option 3 models:
    - 'convlstm': TemporalConvLSTM_UNet with 2D ResNet + ConvLSTM feature cells.
    - 'attention': TemporalAttentionUNet with 2D ResNet + Temporal Self-Attention.
    """
    if model_type == "convlstm":
        base_model = TemporalConvLSTM_UNet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            encoder_weights=encoder_weights
        )
    elif model_type == "attention":
        base_model = TemporalAttentionUNet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            encoder_weights=encoder_weights
        )
    else:
        raise ValueError(f"Unknown sequential model_type: {model_type}")

    return AutoPadWrapper(base_model, multiple=32)
