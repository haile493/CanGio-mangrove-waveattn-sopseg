"""
SOPSeg: Prompt-based Small Object Instance Segmentation in Remote Sensing Imagery
Adapted for Can Gio Mangrove Canopy Disturbance Segmentation (Cut and Anhgoc Imagery).
Reference: Wang et al., 2025/2026 (ref/SOPSeg.pdf)
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam, MaskDecoder
from segment_anything.modeling.mask_decoder import MLP


# ---------------------------------------------------------------------------
# 1. Region-Adaptive Magnification (RAM) Strategy
# ---------------------------------------------------------------------------

class RegionAdaptiveMagnification:
    """
    Region-Adaptive Magnification Strategy (Section 3.1, Eq. 1 & 2 in SOPSeg).
    Adapts window size S based on object dimension d = max(w, h), resizing
    the extracted region to a fixed Sin x Sin (default 256x256).
    """
    def __init__(
        self,
        m: int = 32,
        k0: float = 2.0,
        s_max: int = 1024,
        s_min: int = 16,
        s_in: int = 256
    ):
        self.m = m
        self.k0 = k0
        self.s_max = s_max
        self.s_min = s_min
        self.s_in = s_in
        self.k = (s_max - k0 * m) / float(s_max - m)

    def compute_region_size(self, d: float) -> int:
        """Computes region size S according to Eq. (1)."""
        if d < self.m:
            s = self.k0 * d
        else:
            s = self.k * d + (self.k0 - self.k) * self.m
        s = max(self.s_min, min(self.s_max, int(round(s))))
        return s

    def extract_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Extracts an adaptively magnified square patch around the target object.
        
        Args:
            image: (C, H, W) or (H, W, C) numpy array
            bbox: (x, y, w, h) bounding box of target instance
            mask: (H, W) optional binary instance mask
            is_training: if True, applies random center jitter ax, ay in [0.3, 0.7]
            
        Returns:
            dict containing magnified image (s_in, s_in), magnified mask,
            crop bounds (xs, ys, S, S), and coordinate transformation metadata.
        """
        if image.ndim == 3 and (image.shape[0] < min(image.shape[1], image.shape[2]) or image.shape[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]):
            # Convert (C, H, W) to (H, W, C) for cv2 processing
            c, h, w = image.shape
            img_hwc = np.transpose(image, (1, 2, 0))
            is_chw = True
        else:
            h, w = image.shape[:2]
            img_hwc = image
            is_chw = False

        x, y, bw, bh = bbox
        d = max(bw, bh, 1)
        s = self.compute_region_size(d)

        # Center jitter (Eq. 2)
        if is_training:
            ax = float(np.random.uniform(0.3, 0.7))
            ay = float(np.random.uniform(0.3, 0.7))
        else:
            ax = 0.5
            ay = 0.5

        xs = int(round(x - ax * (s - bw)))
        ys = int(round(y - ay * (s - bh)))

        # Determine padding if window extends outside image boundaries
        pad_left = max(0, -xs)
        pad_top = max(0, -ys)
        pad_right = max(0, (xs + s) - w)
        pad_bottom = max(0, (ys + s) - h)

        src_x1 = max(0, xs)
        src_y1 = max(0, ys)
        src_x2 = min(w, xs + s)
        src_y2 = min(h, ys + s)

        # Slice crop
        crop_img = img_hwc[src_y1:src_y2, src_x1:src_x2]
        
        # Apply padding if necessary
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop_img = cv2.copyMakeBorder(
                crop_img, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REFLECT_101
            )

        # Ensure exact S x S
        if crop_img.shape[0] != s or crop_img.shape[1] != s:
            crop_img = cv2.resize(crop_img, (s, s), interpolation=cv2.INTER_LINEAR)

        # Resize to fixed Sin x Sin (256x256)
        patch_img = cv2.resize(crop_img, (self.s_in, self.s_in), interpolation=cv2.INTER_LINEAR)
        if is_chw:
            patch_img = np.transpose(patch_img, (2, 0, 1))

        # Process mask if provided
        patch_mask = None
        if mask is not None:
            mask_crop = mask[src_y1:src_y2, src_x1:src_x2]
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                mask_crop = cv2.copyMakeBorder(
                    mask_crop, pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_CONSTANT, value=0
                )
            if mask_crop.shape[0] != s or mask_crop.shape[1] != s:
                mask_crop = cv2.resize(mask_crop.astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
            patch_mask = cv2.resize(mask_crop.astype(np.uint8), (self.s_in, self.s_in), interpolation=cv2.INTER_NEAREST)

        meta = {
            "orig_size": (h, w),
            "crop_bounds": (xs, ys, s, s),
            "pad": (pad_left, pad_top, pad_right, pad_bottom),
            "scale_to_patch": float(self.s_in / s),
            "d": d,
            "S": s
        }

        return {
            "patch_img": patch_img,
            "patch_mask": patch_mask,
            "meta": meta
        }

    def inverse_map_patch_to_canvas(
        self,
        patch_prob: np.ndarray,
        meta: Dict[str, Any],
        canvas: np.ndarray
    ) -> np.ndarray:
        """
        Projects 256x256 patch prediction probabilities back to the full scene canvas.
        """
        h_orig, w_orig = meta["orig_size"]
        xs, ys, s, _ = meta["crop_bounds"]
        pad_left, pad_top, pad_right, pad_bottom = meta["pad"]

        # Downsample patch back to crop size s x s using bilinear interpolation
        re_s = cv2.resize(patch_prob.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)

        h_unpad = s - pad_top - pad_bottom
        w_unpad = s - pad_left - pad_right
        unpadded = re_s[pad_top : pad_top + h_unpad, pad_left : pad_left + w_unpad]

        src_x1 = max(0, xs)
        src_y1 = max(0, ys)
        src_x2 = min(w_orig, xs + s)
        src_y2 = min(h_orig, ys + s)

        # Blend with existing canvas using maximum probability
        canvas[src_y1:src_y2, src_x1:src_x2] = np.maximum(
            canvas[src_y1:src_y2, src_x1:src_x2], unpadded
        )
        return canvas

    def map_back_to_scene(
        self,
        patch_prob: np.ndarray,
        orig_shape: Tuple[int, int],
        meta: Dict[str, Any]
    ) -> np.ndarray:
        """Convenience method to map patch back to empty or existing scene canvas."""
        canvas = np.zeros(orig_shape, dtype=np.float32)
        return self.inverse_map_patch_to_canvas(patch_prob, meta, canvas)


# ---------------------------------------------------------------------------
# 2. Oriented Prompt Mechanism (OPM)
# ---------------------------------------------------------------------------

class OrientedPromptMechanism:
    """
    Oriented Prompt Mechanism (Section 3.2, Eq. 4, 5, 6 in SOPSeg).
    Extracts geometric center C and 2 directional keypoints P1, P2 along the
    object's principal axis, combined with a horizontal bounding box.
    """
    @staticmethod
    def extract_oriented_prompts(
        instance_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts:
        - points: (3, 2) array of (x, y) coordinates for [P1, C, P2]
        - point_labels: (3,) array with foreground label [1, 1, 1]
        - box: (4,) horizontal bounding box [x_min, y_min, x_max, y_max]
        """
        bin_mask = (instance_mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = instance_mask.shape
        if not contours:
            # Fallback for empty mask: center point and unit box
            cx, cy = w / 2.0, h / 2.0
            pts = np.array([[cx, cy], [cx, cy], [cx, cy]], dtype=np.float32)
            lbls = np.array([1, 1, 1], dtype=np.int32)
            box = np.array([cx - 2, cy - 2, cx + 2, cy + 2], dtype=np.float32)
            return pts, lbls, box

        # Select largest contour
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            # Axis-aligned fallback
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            p1 = [cx, by + bh * 0.25]
            p2 = [cx, by + bh * 0.75]
            pts = np.array([p1, [cx, cy], p2], dtype=np.float32)
            lbls = np.array([1, 1, 1], dtype=np.int32)
            box = np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
            return pts, lbls, box

        # Oriented bounding box via minAreaRect
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        box_pts = cv2.boxPoints(rect)  # 4 vertices

        # Order box vertices and find shorter sides
        # Distance between adjacent vertices
        d01 = np.linalg.norm(box_pts[0] - box_pts[1])
        d12 = np.linalg.norm(box_pts[1] - box_pts[2])

        if d01 <= d12:
            # Shorter sides are (0, 1) and (2, 3)
            c1 = (box_pts[0] + box_pts[1]) / 2.0
            c2 = (box_pts[2] + box_pts[3]) / 2.0
        else:
            # Shorter sides are (1, 2) and (3, 0)
            c1 = (box_pts[1] + box_pts[2]) / 2.0
            c2 = (box_pts[3] + box_pts[0]) / 2.0

        # Principal directional points (Eq. 4)
        c = np.array([cx, cy], dtype=np.float32)
        p1 = (c + c1) / 2.0
        p2 = (c + c2) / 2.0

        pts = np.array([p1, c, p2], dtype=np.float32)
        lbls = np.array([1, 1, 1], dtype=np.int32)

        # Horizontal bounding box
        bx, by, bw, bh = cv2.boundingRect(cnt)
        box = np.array([bx, by, bx + bw, by + bh], dtype=np.float32)

        return pts, lbls, box

    @staticmethod
    def transform_prompts_to_patch(
        pts: np.ndarray,
        box: np.ndarray,
        meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms original image coordinates of points and box to patch coordinates."""
        xs, ys, s, _ = meta["crop_bounds"]
        scale = meta["scale_to_patch"]
        s_in = meta.get("s_in", 256)

        patch_pts = pts.copy()
        patch_pts[:, 0] = (patch_pts[:, 0] - xs) * scale
        patch_pts[:, 1] = (patch_pts[:, 1] - ys) * scale
        patch_pts = np.clip(patch_pts, 0, s_in - 1)

        patch_box = box.copy()
        patch_box[0] = (patch_box[0] - xs) * scale
        patch_box[1] = (patch_box[1] - ys) * scale
        patch_box[2] = (patch_box[2] - xs) * scale
        patch_box[3] = (patch_box[3] - ys) * scale
        patch_box = np.clip(patch_box, 0, s_in - 1)

        return patch_pts, patch_box


# ---------------------------------------------------------------------------
# 3. Position Embedding Interpolation (PEI)
# ---------------------------------------------------------------------------

def apply_position_embedding_interpolation(sam_model: Sam, target_size: int = 256):
    """
    Interpolates SAM ViT-B image encoder positional embeddings to support
    target input size (Section 3.1, Eq. 3 in SOPSeg).
    Default SAM: 1024x1024 input -> 64x64 tokens.
    Target: 256x256 input -> 16x16 tokens.
    """
    target_grid = target_size // 16
    curr_pe = sam_model.image_encoder.pos_embed.data  # (1, H, W, C)
    if curr_pe.shape[1] != target_grid or curr_pe.shape[2] != target_grid:
        pe_perm = curr_pe.permute(0, 3, 1, 2)  # (1, C, H, W)
        interp_pe = F.interpolate(pe_perm, size=(target_grid, target_grid), mode="bilinear", align_corners=False)
        interp_pe = interp_pe.permute(0, 2, 3, 1)  # (1, target_grid, target_grid, C)
        sam_model.image_encoder.pos_embed = nn.Parameter(interp_pe, requires_grad=False)
        print(f"[SOPSeg PEI] Positional embeddings interpolated to ({target_grid}x{target_grid}).")

    # Update prompt encoder embedding grid size
    sam_model.prompt_encoder.image_embedding_size = (target_grid, target_grid)
    sam_model.prompt_encoder.input_image_size = (target_size, target_size)


# ---------------------------------------------------------------------------
# 4. Enhanced Decoder with Edge Prediction & Progressive Refinement
# ---------------------------------------------------------------------------

class ResidualRefinementBlock(nn.Module):
    """
    Residual Refinement Block (R_j) for multi-scale progressive enhancement (Eq. 9, 11).
    Fuses previous feature representations, image guidance, and intermediate predictions.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + res)
        return out


class ProgressiveRefinementModule(nn.Module):
    """
    Multi-Scale Progressive Refinement Module (Section 3.3, Stage 2 in SOPSeg).
    Operates across spatial scales: 1/8 -> 1/4 -> 1/2 -> 1/1.
    """
    def __init__(
        self,
        shallow_dim: int = 768,
        deep_dim: int = 256,
        feat_dim: int = 64
    ):
        super().__init__()
        # 1. Map and upsample shallow (Fs) and deep (Fd) features from 1/16 to 1/8
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

        # Stage 1/8 -> 1/4: input = [Fs_8; Fd_8; I_8; P_8] -> 64 + 64 + 3 + 2 = 133 channels
        self.r1 = nn.Sequential(
            ResidualRefinementBlock(feat_dim * 2 + 3 + 2, feat_dim),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.phi1 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M4, E4] at 1/4 scale (64x64)

        # Stage 1/4 -> 1/2: input = [X4; I4; P4] -> 64 + 3 + 2 = 69 channels
        self.r2 = nn.Sequential(
            ResidualRefinementBlock(feat_dim + 3 + 2, feat_dim),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.phi2 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M2, E2] at 1/2 scale (128x128)

        # Stage 1/2 -> 1/1: input = [X2; I2; P2] -> 64 + 3 + 2 = 69 channels
        self.r3 = nn.Sequential(
            ResidualRefinementBlock(feat_dim + 3 + 2, feat_dim),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )
        self.phi3 = nn.Conv2d(feat_dim, 2, kernel_size=1)  # [M1, E1] at 1/1 scale (256x256)

        # Mask quality / IoU prediction head (Eq. 12)
        self.iou_head = nn.Sequential(
            nn.Conv2d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        shallow_feats: torch.Tensor,   # Fs (B, 768, 16, 16)
        deep_feats: torch.Tensor,      # Fd (B, 256, 16, 16)
        image_patch: torch.Tensor,     # I  (B, 3, 256, 256)
        init_preds: torch.Tensor       # P0 [M0; E0] (B, 2, 64, 64)
    ) -> Dict[str, torch.Tensor]:
        b = image_patch.shape[0]

        # 1. Scale 1/8 features
        f_s8 = self.map_shallow(shallow_feats)  # (B, 64, 32, 32)
        f_d8 = self.map_deep(deep_feats)        # (B, 64, 32, 32)
        i8 = F.interpolate(image_patch, size=(32, 32), mode="bilinear", align_corners=False)
        p8 = F.interpolate(init_preds, size=(32, 32), mode="bilinear", align_corners=False)

        # Refinement to 1/4 (64x64)
        cat8 = torch.cat([f_s8, f_d8, i8, p8], dim=1)
        x4 = self.r1(cat8)                     # (B, 64, 64, 64)
        p4 = self.phi1(x4)                     # (B, 2, 64, 64)

        # Refinement to 1/2 (128x128)
        i4 = F.interpolate(image_patch, size=(64, 64), mode="bilinear", align_corners=False)
        cat4 = torch.cat([x4, i4, p4], dim=1)
        x2 = self.r2(cat4)                     # (B, 64, 128, 128)
        p2 = self.phi2(x2)                     # (B, 2, 128, 128)

        # Refinement to 1/1 (256x256)
        i2 = F.interpolate(image_patch, size=(128, 128), mode="bilinear", align_corners=False)
        cat2 = torch.cat([x2, i2, p2], dim=1)
        x1 = self.r3(cat2)                     # (B, 64, 256, 256)
        p1 = self.phi3(x1)                     # (B, 2, 256, 256)

        # IoU score prediction
        iou_score = self.iou_head(x1)          # (B, 1)

        return {
            "p4": p4,                          # (B, 2, 64, 64)   [M4, E4]
            "p2": p2,                          # (B, 2, 128, 128) [M2, E2]
            "p1": p1,                          # (B, 2, 256, 256) [M1, E1]
            "iou_pred": iou_score              # (B, 1)
        }


class SOPSegMaskDecoder(MaskDecoder):
    """
    Enhanced Mask Decoder with Learnable Edge Token (Section 3.3, Stage 1 in SOPSeg).
    Augments SAM's MaskDecoder with a learnable edge token T_edge and parallel
    MLP head for edge boundary prediction.
    """
    def __init__(self, base_decoder: MaskDecoder):
        super().__init__(
            transformer_dim=base_decoder.transformer_dim,
            transformer=base_decoder.transformer,
            num_multimask_outputs=base_decoder.num_mask_tokens - 1,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256
        )
        # Copy pretrained weights
        self.load_state_dict(base_decoder.state_dict(), strict=False)

        # Learnable edge token T_edge (Eq. 7, 8)
        self.edge_token = nn.Embedding(1, self.transformer_dim)
        # Edge prediction MLP matching mask prediction MLP
        self.output_hypernetworks_mlp_edge = MLP(
            self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts initial coarse mask M0 and edge E0 along with decoder deep feature Fd.
        """
        # Concatenate: [iou_token, mask_tokens, edge_token]
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.edge_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand image embeddings if needed
        if image_embeddings.shape[0] == tokens.shape[0]:
            src = image_embeddings
        else:
            repeats = tokens.shape[0] // image_embeddings.shape[0]
            src = torch.repeat_interleave(image_embeddings, repeats, dim=0)
        src = src + dense_prompt_embeddings

        if image_pe.shape[0] == tokens.shape[0]:
            pos_src = image_pe
        else:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Two-way transformer cross-attention
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_token_out = hs[:, 1, :]         # Single mask token for binary target
        edge_token_out = hs[:, 1 + self.num_mask_tokens, :]  # Edge token

        # Deep feature Fd
        fd = src.transpose(1, 2).view(b, c, h, w)  # (B, 256, 16, 16)
        upscaled_embedding = self.output_upscaling(fd)  # (B, 32, 64, 64)

        # Mask prediction M0 (Eq. 7)
        mask_hyper = self.output_hypernetworks_mlps[0](mask_token_out)  # (B, 32)
        b_up, c_up, h_up, w_up = upscaled_embedding.shape
        m0 = (mask_hyper.unsqueeze(1) @ upscaled_embedding.view(b_up, c_up, h_up * w_up)).view(b_up, 1, h_up, w_up)

        # Edge prediction E0 (Eq. 8)
        edge_hyper = self.output_hypernetworks_mlp_edge(edge_token_out)  # (B, 32)
        e0 = (edge_hyper.unsqueeze(1) @ upscaled_embedding.view(b_up, c_up, h_up * w_up)).view(b_up, 1, h_up, w_up)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return m0, e0, fd, iou_pred


# ---------------------------------------------------------------------------
# 5. Complete SOPSeg Model
# ---------------------------------------------------------------------------

class SOPSegModel(nn.Module):
    """
    Complete SOPSeg Framework integrating:
    1. SAM ViT Image Encoder with Position Embedding Interpolation (PEI)
    2. SAM Prompt Encoder (Box + Oriented Keypoints)
    3. Enhanced Edge Decoder (Stage 1: M0, E0, Fd)
    4. Multi-Scale Progressive Refinement (Stage 2: P4, P2, P1, IoU)
    """
    def __init__(
        self,
        base_sam: Sam,
        shallow_layer_idx: int = 3,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0
    ):
        super().__init__()
        self.image_encoder = base_sam.image_encoder
        self.prompt_encoder = base_sam.prompt_encoder
        self.shallow_layer_idx = shallow_layer_idx
        self.use_lora = use_lora

        if use_lora:
            from src.sam_satellite import apply_lora_to_sam
            apply_lora_to_sam(base_sam, rank=lora_rank, alpha=lora_alpha)
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False
        else:
            # Freeze image and prompt encoders as per vanilla paper
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False

        # Enhanced Decoder with Edge Prediction
        self.mask_decoder = SOPSegMaskDecoder(base_sam.mask_decoder)

        # Progressive Refinement Module
        shallow_dim = getattr(base_sam.image_encoder, "embed_dim", base_sam.image_encoder.patch_embed.proj.out_channels)
        self.refinement = ProgressiveRefinementModule(
            shallow_dim=shallow_dim,
            deep_dim=base_sam.mask_decoder.transformer_dim,
            feat_dim=64
        )

    def extract_shallow_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts shallow feature representations from early ViT block."""
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)
            if i == self.shallow_layer_idx:
                # Shape is (B, H, W, C) -> permute to (B, C, H, W)
                return x.permute(0, 3, 1, 2)
        return x.permute(0, 3, 1, 2)

    def forward(
        self,
        image_patch: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of SOPSeg on an adaptively magnified patch (256x256).
        """
        # 1. Extract shallow features Fs
        shallow_feats = self.extract_shallow_features(image_patch)

        # 2. Extract full image embeddings from frozen ViT
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

        # 5. Stage 2: Progressive Multi-Scale Refinement
        p0 = torch.cat([m0, e0], dim=1)  # (B, 2, 64, 64)
        refined = self.refinement(
            shallow_feats=shallow_feats,
            deep_feats=fd,
            image_patch=image_patch,
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
# 6. Multi-Task Loss Function
# ---------------------------------------------------------------------------

class SOPSegMultiTaskLoss(nn.Module):
    """
    Multi-task loss for SOPSeg (Section 3.3, Eq. 13 in SOPSeg):
    L = sum_{i in {1, 2, 4}} (L_mask^i + L_edge^i) + lambda_iou * L_iou
    where L_mask and L_edge are BCE + Dice losses at scale i.
    """
    def __init__(self, lambda_iou: float = 5.0):
        super().__init__()
        self.lambda_iou = lambda_iou
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    @staticmethod
    def dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(pred_logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = (2.0 * intersection + eps) / (union + eps)
        return (1.0 - dice).mean()

    @staticmethod
    def extract_gt_edges(gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Derives smoothed ground-truth edge map using Sobel filters and 3x3 Gaussian smoothing.
        gt_masks: (B, 1, H, W) binary float tensor
        """
        device = gt_masks.device
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

        gx = F.conv2d(gt_masks, sobel_x, padding=1)
        gy = F.conv2d(gt_masks, sobel_y, padding=1)
        edges = torch.sqrt(gx**2 + gy**2 + 1e-8)
        edges = (edges > 0.1).float()

        # Gaussian smoothing kernel 3x3
        gauss_k = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 16.0
        smoothed = F.conv2d(edges, gauss_k, padding=1)
        return torch.clamp(smoothed, 0.0, 1.0)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        predictions: dict with p4 (64x64), p2 (128x128), p1 (256x256), iou_pred
        gt_mask: (B, 1, 256, 256) float tensor in [0, 1]
        """
        gt_edge = self.extract_gt_edges(gt_mask)

        total_loss = 0.0
        scale_losses = {}

        # Scale mappings: p1 -> 256, p2 -> 128, p4 -> 64
        scales = {
            "scale_1": (predictions["p1"], 256),
            "scale_2": (predictions["p2"], 128),
            "scale_4": (predictions["p4"], 64)
        }

        actual_iou = None

        for s_name, (pred_tensor, size) in scales.items():
            mask_logits = pred_tensor[:, 0:1, :, :]
            edge_logits = pred_tensor[:, 1:2, :, :]

            gt_m_scale = F.interpolate(gt_mask, size=(size, size), mode="bilinear", align_corners=False)
            gt_e_scale = F.interpolate(gt_edge, size=(size, size), mode="bilinear", align_corners=False)

            l_mask = self.bce(mask_logits, gt_m_scale) + self.dice_loss(mask_logits, gt_m_scale)
            l_edge = self.bce(edge_logits, gt_e_scale) + self.dice_loss(edge_logits, gt_e_scale)

            loss_s = l_mask + l_edge
            total_loss = total_loss + loss_s
            scale_losses[f"{s_name}_mask"] = l_mask.item()
            scale_losses[f"{s_name}_edge"] = l_edge.item()

            if size == 256:
                # Compute actual IoU for quality head
                with torch.no_grad():
                    pred_bin = (torch.sigmoid(mask_logits) >= 0.5).float()
                    inter = (pred_bin * gt_m_scale).sum(dim=(-2, -1))
                    union = pred_bin.sum(dim=(-2, -1)) + gt_m_scale.sum(dim=(-2, -1)) - inter
                    actual_iou = (inter / (union + 1e-6)).view(-1, 1)

        # IoU Loss (Smooth L1)
        l_iou = self.smooth_l1(predictions["iou_pred"], actual_iou)
        total_loss = total_loss + self.lambda_iou * l_iou
        scale_losses["iou_loss"] = l_iou.item()
        scale_losses["total_loss"] = total_loss

        return scale_losses


# ---------------------------------------------------------------------------
# 7. Unit Test Runner
# ---------------------------------------------------------------------------

def test_sopseg_components():
    """Verifies all components of SOPSeg framework."""
    print("Testing SOPSeg Components...")

    # 1. Test RAM
    ram = RegionAdaptiveMagnification(s_in=256)
    dummy_img = np.random.randint(0, 255, (124, 124, 3), dtype=np.uint8)
    dummy_mask = np.zeros((124, 124), dtype=np.uint8)
    dummy_mask[50:60, 50:58] = 1
    res = ram.extract_region(dummy_img, (50, 50, 8, 10), mask=dummy_mask, is_training=False)
    assert res["patch_img"].shape == (256, 256, 3), f"RAM patch_img shape mismatch: {res['patch_img'].shape}"
    assert res["patch_mask"].shape == (256, 256), f"RAM patch_mask shape mismatch: {res['patch_mask'].shape}"
    print("  [PASS] RegionAdaptiveMagnification verified.")

    # 2. Test OPM
    opm = OrientedPromptMechanism()
    pts, lbls, box = opm.extract_oriented_prompts(dummy_mask)
    assert pts.shape == (3, 2), f"OPM points shape mismatch: {pts.shape}"
    assert box.shape == (4,), f"OPM box shape mismatch: {box.shape}"
    p_pts, p_box = opm.transform_prompts_to_patch(pts, box, res["meta"])
    assert p_pts.shape == (3, 2)
    print("  [PASS] OrientedPromptMechanism verified.")

    # 3. Test Loss
    loss_fn = SOPSegMultiTaskLoss()
    dummy_gt = torch.zeros(2, 1, 256, 256)
    dummy_gt[:, :, 100:150, 100:150] = 1.0
    dummy_preds = {
        "p4": torch.randn(2, 2, 64, 64),
        "p2": torch.randn(2, 2, 128, 128),
        "p1": torch.randn(2, 2, 256, 256),
        "iou_pred": torch.tensor([[0.8], [0.7]])
    }
    losses = loss_fn(dummy_preds, dummy_gt)
    assert "total_loss" in losses
    print(f"  [PASS] SOPSegMultiTaskLoss verified (Loss = {losses['total_loss'].item():.4f}).")

    print("[ALL PASSED] SOPSeg unit tests succeeded.")


if __name__ == "__main__":
    test_sopseg_components()
