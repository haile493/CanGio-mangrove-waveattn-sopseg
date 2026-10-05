"""
Bi-Temporal Siamese SOPSeg-CD: Small Object Change Detection Framework
Integrating SOPSeg (RAM, OPM, EDE) + LoRA with BiSAM-CD & SAM2-CD.

Key Components:
1. Synchronized Bi-Temporal RAM: Synchronously extracts adaptively magnified patches
   (256x256) from both T1 (pre-event) and T2 (post-event) satellite scenes.
2. Siamese SAM ViT-B + LoRA Encoder: Weight-shared ViT-B encoder extracting multi-level
   features for both timestamps with parameter-efficient rank-8 LoRA adaptation.
3. Temporal Difference & Fusion Module (TFM / ASG from SAM2-CD):
   Computes cross-temporal differences |F_T2 - F_T1| with Activation Selection Gating.
4. Bi-Temporal WaveFusion Refinement:
   Refines change masks and change boundary edges across 1/8 -> 1/4 -> 1/2 -> 1/1 scales
   using Haar Wavelet decomposition and bi-temporal spectral delta guidance.
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam, MaskDecoder

from src.sopseg import (
    RegionAdaptiveMagnification,
    OrientedPromptMechanism,
    apply_position_embedding_interpolation,
    SOPSegMaskDecoder,
    SOPSegMultiTaskLoss
)
from src.sopseg_enhanced import (
    HaarDWT2D,
    HaarIDWT2D,
    WaveFusionRefinementBlock,
    GLCAMContextAttention
)
from src.sam_satellite import apply_lora_to_sam


# ---------------------------------------------------------------------------
# 1. Temporal Difference & Fusion Module (SAM2-CD ASG / TFM)
# ---------------------------------------------------------------------------

class TemporalFusionModule(nn.Module):
    """
    Fuses bi-temporal feature representations using absolute difference
    and activation selection gating (ASG) from SAM2-CD.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Absolute difference and concatenated projection: [F1, F2, |F2 - F1|] -> in_channels * 3
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Activation Selection Gate (ASG): learns channel-wise significance of changes
        self.asg_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(1, out_channels // 8), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_channels // 8), out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f2 - f1)
        cat = torch.cat([f1, f2, diff], dim=1)
        fused = self.fuse_conv(cat)
        gate = self.asg_gate(fused)
        out = fused * gate + fused  # Gated residual
        return out


# ---------------------------------------------------------------------------
# 2. Bi-Temporal Progressive Refinement Module
# ---------------------------------------------------------------------------

class BiTemporalRefinementModule(nn.Module):
    """
    Refines change masks and change edges across scales (1/8 -> 1/4 -> 1/2 -> 1/1)
    fusing temporal difference features, bi-temporal image guidance, and intermediate predictions.
    """
    def __init__(
        self,
        shallow_dim: int = 768,
        deep_dim: int = 256,
        feat_dim: int = 64,
        guidance_channels: int = 7  # 3 (T1) + 3 (T2) + 1 (delta NBR)
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.guidance_channels = guidance_channels

        # Attention on temporal difference features
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

        # Stage 1/8 -> 1/4: input = [Fs_diff_8; Fd_diff_8; I_bitemp_8; P_8]
        in_c1 = feat_dim * 2 + guidance_channels + 2
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_c1, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.wave1 = WaveFusionRefinementBlock(feat_dim, feat_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.phi1 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M4_diff, E4_diff]

        # Stage 1/4 -> 1/2: input = [X4; I_bitemp_4; P4]
        in_c2 = feat_dim + guidance_channels + 2
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_c2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.wave2 = WaveFusionRefinementBlock(feat_dim, feat_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.phi2 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M2_diff, E2_diff]

        # Stage 1/2 -> 1/1: input = [X2; I_bitemp_2; P2]
        in_c3 = feat_dim + guidance_channels + 2
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_c3, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.wave3 = WaveFusionRefinementBlock(feat_dim, feat_dim)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.phi3 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M1_diff, E1_diff]

        # Quality head for change mask IoU
        self.iou_head = nn.Sequential(
            nn.Conv2d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        shallow_diff: torch.Tensor,       # (B, 768, 16, 16)
        deep_diff: torch.Tensor,          # (B, 256, 16, 16)
        bitemp_guidance: torch.Tensor,    # (B, 7, 256, 256)
        init_preds: torch.Tensor          # P0 [M0; E0] (B, 2, 64, 64)
    ) -> Dict[str, torch.Tensor]:
        s_att = self.glcam_shallow(shallow_diff)
        d_att = self.glcam_deep(deep_diff)

        f_s8 = self.map_shallow(s_att)
        f_d8 = self.map_deep(d_att)
        i8 = F.interpolate(bitemp_guidance, size=(32, 32), mode="bilinear", align_corners=False)
        p8 = F.interpolate(init_preds, size=(32, 32), mode="bilinear", align_corners=False)

        cat8 = torch.cat([f_s8, f_d8, i8, p8], dim=1)
        feat8 = self.fuse1(cat8)
        x4 = self.up1(self.wave1(feat8))
        p4 = self.phi1(x4)

        i4 = F.interpolate(bitemp_guidance, size=(64, 64), mode="bilinear", align_corners=False)
        cat4 = torch.cat([x4, i4, p4], dim=1)
        feat4 = self.fuse2(cat4)
        x2 = self.up2(self.wave2(feat4))
        p2 = self.phi2(x2)

        i2 = F.interpolate(bitemp_guidance, size=(128, 128), mode="bilinear", align_corners=False)
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
# 3. Complete BiTemporalSOPSegModel (Option 2)
# ---------------------------------------------------------------------------

class BiTemporalSOPSegModel(nn.Module):
    """
    Bi-Temporal Siamese SOPSeg Change Detection Model.
    Processes (T1, T2) pairs with shared ViT-B + LoRA, fuses temporal difference
    features via ASG, and performs progressive WaveFusion refinement.
    """
    def __init__(
        self,
        base_sam: Sam,
        shallow_layer_idx: int = 3,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0
    ):
        super().__init__()
        self.image_encoder = base_sam.image_encoder
        self.prompt_encoder = base_sam.prompt_encoder
        self.shallow_layer_idx = shallow_layer_idx
        self.use_lora = use_lora

        if use_lora:
            apply_lora_to_sam(base_sam, rank=lora_rank, alpha=lora_alpha)
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False
        else:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False

        shallow_dim = getattr(base_sam.image_encoder, "embed_dim", base_sam.image_encoder.patch_embed.proj.out_channels)
        deep_dim = base_sam.mask_decoder.transformer_dim

        # Temporal Fusion Modules for shallow and deep ViT features
        self.tfm_shallow = TemporalFusionModule(shallow_dim, shallow_dim)
        self.tfm_deep = TemporalFusionModule(deep_dim, deep_dim)

        # Enhanced Edge Decoder for change masks
        self.mask_decoder = SOPSegMaskDecoder(base_sam.mask_decoder)

        # Bi-Temporal WaveFusion Refinement Module
        self.refinement = BiTemporalRefinementModule(
            shallow_dim=shallow_dim,
            deep_dim=deep_dim,
            feat_dim=64,
            guidance_channels=7
        )

    def extract_shallow_features(self, x: torch.Tensor) -> torch.Tensor:
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
        patch_t1: torch.Tensor,
        patch_t2: torch.Tensor,
        bitemp_guidance: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for bi-temporal change detection pair.
        patch_t1: (B, 3, 256, 256)
        patch_t2: (B, 3, 256, 256)
        bitemp_guidance: (B, 7, 256, 256) [T1; T2; delta_NBR]
        """
        # 1. Siamese ViT Feature Extraction
        fs1 = self.extract_shallow_features(patch_t1)
        fs2 = self.extract_shallow_features(patch_t2)

        fd1 = self.image_encoder(patch_t1)
        fd2 = self.image_encoder(patch_t2)

        # 2. Temporal Fusion & Activation Selection Gating
        fs_diff = self.tfm_shallow(fs1, fs2)
        fd_diff = self.tfm_deep(fd1, fd2)

        # 3. Prompt Encoder
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None
        )

        # 4. Stage 1: Decoder conditioned on temporal difference embeddings
        m0, e0, fd_out, init_iou = self.mask_decoder(
            image_embeddings=fd_diff,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings
        )

        # 5. Stage 2: Bi-Temporal WaveFusion Refinement
        p0 = torch.cat([m0, e0], dim=1)
        refined = self.refinement(
            shallow_diff=fs_diff,
            deep_diff=fd_out,
            bitemp_guidance=bitemp_guidance,
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


# ---------------------------------------------------------------------------
# 4. Unit Test Runner
# ---------------------------------------------------------------------------

def test_bitemporal_sopseg():
    print("Testing BiTemporal SOPSeg components...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = TemporalFusionModule(64, 64).to(device)
    f1 = torch.randn(2, 64, 16, 16, device=device)
    f2 = torch.randn(2, 64, 16, 16, device=device)
    out = tfm(f1, f2)
    assert out.shape == (2, 64, 16, 16)
    print("  [PASS] TemporalFusionModule (ASG) verified.")

    refine = BiTemporalRefinementModule(shallow_dim=64, deep_dim=64, feat_dim=64, guidance_channels=7).to(device)
    g = torch.randn(2, 7, 256, 256, device=device)
    p0 = torch.randn(2, 2, 64, 64, device=device)
    res = refine(out, out, g, p0)
    assert res["p1"].shape == (2, 2, 256, 256)
    assert res["iou_pred"].shape == (2, 1)
    print("  [PASS] BiTemporalRefinementModule verified.")

    print("[ALL PASSED] BiTemporal SOPSeg unit tests succeeded.")


if __name__ == "__main__":
    test_bitemporal_sopseg()
