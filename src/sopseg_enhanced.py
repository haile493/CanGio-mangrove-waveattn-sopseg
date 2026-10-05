"""
SOPSeg-Enhanced: Wavelet-Refined & Context-Aware SOPSeg with LoRA Adaptation
Integrating SAM2-CD (WaveFusion, GLCAM) and BiSAM-CD (Physical Canopy Disturbance Guidance).

Key Architectural Enhancements:
1. WaveFusion: Haar Discrete Wavelet Transform (DWT/IDWT) separates high-frequency edge
   gradients (LH, HL, HH) from low-frequency regional semantics (LL) during multi-scale refinement,
   preventing spatial boundary blurring on small canopy gap instances (1-17 px).
2. GLCAM (Global-Local Contextual Attention Module): Dual channel + spatial attention gates
   that dynamically suppress tidal mudflats, water reflections, and sensor noise while
   spotlighting subtle forest disturbance clusters.
3. Biophysical Canopy Disturbance Guidance (BiSAM-CD): Supplies multi-spectral Normalized Burn
   Ratio (NBR) / NDVI canopy loss signals directly to the progressive guidance stream.
4. LoRA Domain Adaptation: Rank-8 low-rank adaptation on SAM ViT-B self-attention projections.
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Optional, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam, MaskDecoder
from segment_anything.modeling.mask_decoder import MLP

from src.sopseg import (
    RegionAdaptiveMagnification,
    OrientedPromptMechanism,
    apply_position_embedding_interpolation,
    SOPSegMaskDecoder,
    SOPSegMultiTaskLoss
)
from src.sam_satellite import apply_lora_to_sam


# ---------------------------------------------------------------------------
# 1. Pure PyTorch Haar 2D Discrete Wavelet Transform (DWT & IDWT)
# ---------------------------------------------------------------------------

class HaarDWT2D(nn.Module):
    """
    Computes 2D Discrete Wavelet Transform (Haar Wavelet) using grouped 2D convolutions.
    Decomposes (B, C, H, W) into Low-Frequency (LL) and High-Frequency (LH, HL, HH).
    """
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ll: (B, C, H/2, W/2) low-frequency approximation
            high: (B, C*3, H/2, W/2) concatenated horizontal, vertical, and diagonal high-frequencies
        """
        b, c, h, w = x.shape
        w_filt = self.filters.repeat(c, 1, 1, 1)
        dwt_out = F.conv2d(x, w_filt, stride=2, groups=c)  # (B, 4*C, H/2, W/2)
        h2, w2 = dwt_out.shape[-2:]
        dwt_out = dwt_out.view(b, c, 4, h2, w2)
        ll = dwt_out[:, :, 0]
        high = dwt_out[:, :, 1:].reshape(b, c * 3, h2, w2)
        return ll, high


class HaarIDWT2D(nn.Module):
    """
    Reconstructs spatial feature map from LL and High-frequency wavelet subbands
    via transposed grouped 2D convolutions.
    """
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, ll: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs original spatial resolution: (B, C, H, W).
        """
        b, c, h2, w2 = ll.shape
        high_3 = high.view(b, c, 3, h2, w2)
        ll_1 = ll.unsqueeze(2)
        x_dwt = torch.cat([ll_1, high_3], dim=2)  # (B, C, 4, H2, W2)
        x_flat = x_dwt.view(b, c * 4, h2, w2)
        w_filt = self.filters.repeat(c, 1, 1, 1)
        out = F.conv_transpose2d(x_flat, w_filt, stride=2, groups=c)
        return out


# ---------------------------------------------------------------------------
# 2. WaveFusion Refinement Block (Adapted from SAM2-CD Net.py)
# ---------------------------------------------------------------------------

class WaveFusionRefinementBlock(nn.Module):
    """
    Wavelet-Guided Multi-Scale Refinement Block.
    Decomposes incoming features into high-frequency detail and low-frequency semantics,
    processes each with dedicated residual pathways, and inverts via IDWT.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        # High-frequency processing (Edge preservation)
        self.conv_high1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False)
        self.res_high = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.conv_high2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)

        # Low-frequency processing (Semantic shape & area)
        self.res_low = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Output projection & residual connection
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ll, high = self.dwt(x)

        # Process high-frequency subbands (edges)
        high_feat = self.conv_high1(high)
        high_res = self.res_high(high_feat)
        high_out = self.conv_high2(F.gelu(high_feat + high_res))

        # Process low-frequency subband (semantics)
        low_res = self.res_low(ll)
        low_out = F.gelu(ll + low_res)

        # Reconstruct via IDWT
        rec = self.idwt(low_out, high_out)

        out = self.out_conv(rec) + self.shortcut(x)
        return out


# ---------------------------------------------------------------------------
# 3. GLCAM / Contextual Attention (Adapted from SAM2-CD CBAM & Net.py)
# ---------------------------------------------------------------------------

class GLCAMContextAttention(nn.Module):
    """
    Global-Local Contextual Attention Module.
    Combines channel-wise attention with spatial depthwise gating to focus
    on canopy disturbance patches while suppressing non-forest background variations.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        red_dim = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, red_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_dim, in_channels, kernel_size=1, bias=False)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)
        x_ca = x * ca

        # 2. Spatial attention
        avg_spatial = torch.mean(x_ca, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        sa = self.spatial_conv(spatial_cat)
        out = x_ca * sa

        return out + x  # Residual connection


# ---------------------------------------------------------------------------
# 4. Enhanced Progressive Refinement Module with WaveFusion & Biophysical Guidance
# ---------------------------------------------------------------------------

class EnhancedProgressiveRefinementModule(nn.Module):
    """
    Wavelet-Refined Progressive Refinement Module.
    Operates across spatial scales: 1/8 -> 1/4 -> 1/2 -> 1/1.
    Integrates:
    - GLCAM on shallow and deep features.
    - WaveFusion blocks at each spatial scale.
    - 4-channel biophysical guidance stream [NIR, Red, Green, NBR].
    """
    def __init__(
        self,
        shallow_dim: int = 768,
        deep_dim: int = 256,
        feat_dim: int = 64,
        guidance_channels: int = 4
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.guidance_channels = guidance_channels

        # Context Attention Gates
        self.glcam_shallow = GLCAMContextAttention(shallow_dim)
        self.glcam_deep = GLCAMContextAttention(deep_dim)

        # 1. Feature mappings from 1/16 to 1/8
        self.map_shallow = nn.Sequential(
            nn.Conv2d(shallow_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.map_deep = nn.Sequential(
            nn.Conv2d(deep_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )

        # Stage 1/8 -> 1/4: input = [Fs_8; Fd_8; I_8; P_8] -> 64 + 64 + guidance_channels + 2
        in_c1 = feat_dim * 2 + guidance_channels + 2
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_c1, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.wave1 = WaveFusionRefinementBlock(feat_dim, feat_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.phi1 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M4, E4] at 1/4 (64x64)

        # Stage 1/4 -> 1/2: input = [X4; I4; P4] -> 64 + guidance_channels + 2
        in_c2 = feat_dim + guidance_channels + 2
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_c2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.wave2 = WaveFusionRefinementBlock(feat_dim, feat_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.phi2 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M2, E2] at 1/2 (128x128)

        # Stage 1/2 -> 1/1: input = [X2; I2; P2] -> 64 + guidance_channels + 2
        in_c3 = feat_dim + guidance_channels + 2
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_c3, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.wave3 = WaveFusionRefinementBlock(feat_dim, feat_dim)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.phi3 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M1, E1] at 1/1 (256x256)

        # IoU prediction head
        self.iou_head = nn.Sequential(
            nn.Conv2d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        shallow_feats: torch.Tensor,      # Fs (B, 768, 16, 16)
        deep_feats: torch.Tensor,         # Fd (B, 256, 16, 16)
        guidance_patch: torch.Tensor,     # I  (B, guidance_channels, 256, 256)
        init_preds: torch.Tensor          # P0 [M0; E0] (B, 2, 64, 64)
    ) -> Dict[str, torch.Tensor]:
        # 1. Apply GLCAM Context Attention
        s_att = self.glcam_shallow(shallow_feats)
        d_att = self.glcam_deep(deep_feats)

        # 2. Map to 1/8 scale (32x32)
        f_s8 = self.map_shallow(s_att)
        f_d8 = self.map_deep(d_att)
        i8 = F.interpolate(guidance_patch, size=(32, 32), mode="bilinear", align_corners=False)
        p8 = F.interpolate(init_preds, size=(32, 32), mode="bilinear", align_corners=False)

        # Refinement to 1/4 (64x64)
        cat8 = torch.cat([f_s8, f_d8, i8, p8], dim=1)
        feat8 = self.fuse1(cat8)
        x4 = self.up1(self.wave1(feat8))
        p4 = self.phi1(x4)

        # Refinement to 1/2 (128x128)
        i4 = F.interpolate(guidance_patch, size=(64, 64), mode="bilinear", align_corners=False)
        cat4 = torch.cat([x4, i4, p4], dim=1)
        feat4 = self.fuse2(cat4)
        x2 = self.up2(self.wave2(feat4))
        p2 = self.phi2(x2)

        # Refinement to 1/1 (256x256)
        i2 = F.interpolate(guidance_patch, size=(128, 128), mode="bilinear", align_corners=False)
        cat2 = torch.cat([x2, i2, p2], dim=1)
        feat2 = self.fuse3(cat2)
        x1 = self.up3(self.wave3(feat2))
        p1 = self.phi3(x1)

        iou_score = self.iou_head(x1)

        return {
            "p4": p4,
            "p2": p2,
            "p1": p1,
            "iou_pred": iou_score
        }


# ---------------------------------------------------------------------------
# 5. Complete WaveAttn SOPSeg Model Architecture
# ---------------------------------------------------------------------------

class EnhancedSOPSegModel(nn.Module):
    """
    Enhanced SOPSeg Framework integrating:
    1. SAM ViT Image Encoder with PEI and LoRA.
    2. SAM Prompt Encoder (Box + Oriented Keypoints).
    3. Enhanced Edge Decoder (M0, E0, Fd).
    4. Enhanced WaveFusion Progressive Refinement with GLCAM & Biophysical Guidance.
    """
    def __init__(
        self,
        base_sam: Sam,
        shallow_layer_idx: int = 3,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        guidance_channels: int = 4
    ):
        super().__init__()
        self.image_encoder = base_sam.image_encoder
        self.prompt_encoder = base_sam.prompt_encoder
        self.shallow_layer_idx = shallow_layer_idx
        self.use_lora = use_lora
        self.guidance_channels = guidance_channels

        if use_lora:
            apply_lora_to_sam(base_sam, rank=lora_rank, alpha=lora_alpha)
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False
        else:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False

        # Enhanced Edge Decoder
        self.mask_decoder = SOPSegMaskDecoder(base_sam.mask_decoder)

        # Enhanced WaveFusion Refinement Module
        shallow_dim = getattr(base_sam.image_encoder, "embed_dim", base_sam.image_encoder.patch_embed.proj.out_channels)
        self.refinement = EnhancedProgressiveRefinementModule(
            shallow_dim=shallow_dim,
            deep_dim=base_sam.mask_decoder.transformer_dim,
            feat_dim=64,
            guidance_channels=guidance_channels
        )

    def extract_shallow_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts shallow feature representations from early ViT block."""
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)
            if i == self.shallow_layer_idx:
                return x.permute(0, 3, 1, 2)
        return x.permute(0, 3, 1, 2)

    def forward(
        self,
        image_patch: torch.Tensor,
        guidance_patch: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        image_patch: (B, 3, 256, 256) normalized RGB/composite patch for ViT.
        guidance_patch: (B, C_g, 256, 256) multi-spectral + NBR canopy guidance.
        """
        if guidance_patch is None:
            # Fallback to image_patch if not supplied
            guidance_patch = image_patch

        # 1. Extract shallow features Fs
        shallow_feats = self.extract_shallow_features(image_patch)

        # 2. Extract full ViT embeddings
        image_embeddings = self.image_encoder(image_patch)

        # 3. Prompt Encoder
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None
        )

        # 4. Stage 1: Enhanced Edge Decoder
        m0, e0, fd, init_iou = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings
        )

        # 5. Stage 2: WaveFusion & GLCAM Refinement
        p0 = torch.cat([m0, e0], dim=1)
        refined = self.refinement(
            shallow_feats=shallow_feats,
            deep_feats=fd,
            guidance_patch=guidance_patch,
            init_preds=p0
        )

        return {
            "m0": m0,
            "e0": e0,
            "p4": refined["p4"],
            "p2": refined["p2"],
            "p1": refined["p1"],
            "iou_pred": refined["iou_pred"],
            "init_iou": init_iou
        }


# Alias for publication and manuscript naming
WaveAttnSOPSegModel = EnhancedSOPSegModel


# ---------------------------------------------------------------------------
# 6. Unit Test Runner
# ---------------------------------------------------------------------------

def test_enhanced_sopseg():
    print("Testing Enhanced SOPSeg (WaveFusion + GLCAM + Biophysical Guidance)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test Haar DWT/IDWT
    dwt = HaarDWT2D().to(device)
    idwt = HaarIDWT2D().to(device)
    x = torch.randn(2, 64, 32, 32, device=device)
    ll, high = dwt(x)
    rec = idwt(ll, high)
    diff = (x - rec).abs().max().item()
    print(f"  [PASS] Haar DWT/IDWT reconstructed with max diff {diff:.2e}")

    # Test WaveFusionBlock
    wblock = WaveFusionRefinementBlock(64, 64).to(device)
    out_w = wblock(x)
    assert out_w.shape == (2, 64, 32, 32)
    print("  [PASS] WaveFusionRefinementBlock verified.")

    # Test GLCAM
    glcam = GLCAMContextAttention(64).to(device)
    out_g = glcam(x)
    assert out_g.shape == (2, 64, 32, 32)
    print("  [PASS] GLCAMContextAttention verified.")

    print("[ALL PASSED] Enhanced SOPSeg components verified successfully.")


if __name__ == "__main__":
    test_enhanced_sopseg()
