"""
SinkSAM-Net: Knowledge-Driven Self-Supervised Disturbance Segmentation.
Implementation of Rafaeli et al. (ISPRS Journal of Photogrammetry and Remote Sensing, 2025).
GitHub: https://github.com/osherr1996/SinkSAM

Key Stages:
1. Topographic / Depression Prior: Delineation of closed depressions using hydrological "fill sinks".
2. Prompt Generation: Coordinate-Wise Bounding Box Jittering (CWBJ).
3. SAM Monte Carlo Inference: Ensemble simulations with jittered boxes.
4. Shape-Aware Filtering: Polsby-Popper circularity/compactness index filtering.
5. Target Model: Lightweight EffV2-UNet trained with Weighted Soft Dice (WSDice) + BCE loss.
6. Iterative Self-Training: Knowledge transfer to a prompt-free, fast inference model.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import grey_closing, label
from skimage.measure import regionprops, find_contours
from typing import List, Dict, Tuple, Optional, Any
import segmentation_models_pytorch as smp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import AutoPadModelWrapper


# ---------------------------------------------------------------------------
# Stage 1 & 2: Topographic / Depression Prior Computation
# ---------------------------------------------------------------------------

def compute_closed_depressions(
    image_tensor: torch.Tensor,
    closing_size: int = 9,
    depth_thresh: float = 0.12,
    min_area: int = 3,
    max_area: int = 400
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Computes closed depressions on the canopy disturbance surface.
    Canopy collapse / disturbance creates distinct depressions in canopy moisture (NBR).
    
    Args:
        image_tensor: Multi-spectral tensor (C, H, W)
        closing_size: Kernel size for hydrological fill-sinks (grey_closing)
        depth_thresh: Threshold for depression depth (filled - original)
        min_area: Minimum connected component area in pixels
        max_area: Maximum connected component area in pixels
        
    Returns:
        depression_map: Continuous depression depth raster
        cand_mask: Binary closed depression candidate mask
        boxes: List of bounding boxes [x1, y1, x2, y2]
    """
    img_np = image_tensor.numpy() if isinstance(image_tensor, torch.Tensor) else image_tensor
    c, h, w = img_np.shape
    
    # NBR is index 7 if all indices present, or computed from NIR (idx 3) & SWIR2 (idx 5)
    if c >= 8:
        nbr = img_np[7]
        ndvi = img_np[6]
    elif c >= 6:
        nir = img_np[3]
        swir2 = img_np[5]
        red = img_np[2]
        nbr = (nir - swir2) / (nir + swir2 + 1e-6)
        nbr = np.clip((nbr + 1.0) / 2.0, 0.0, 1.0)
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi = np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0)
    else:
        nir = img_np[0]
        red = img_np[1]
        nbr = (nir - red) / (nir + red + 1e-6)
        nbr = np.clip((nbr + 1.0) / 2.0, 0.0, 1.0)
        ndvi = nbr

    # Canopy surface: healthy trees = high elevation, disturbance = depressions
    canopy_surface = nbr.astype(np.float32)
    filled_surface = grey_closing(canopy_surface, size=(closing_size, closing_size))
    depression_map = filled_surface - canopy_surface

    # Forest vegetation mask (exclude deep open water bodies)
    forest_mask = (ndvi > 0.25)
    cand_mask = ((depression_map > depth_thresh) & forest_mask).astype(np.uint8)

    # Connected component labeling to extract bounding boxes
    labeled, num_features = label(cand_mask)
    regions = [r for r in regionprops(labeled) if min_area <= r.area <= max_area]

    boxes = []
    margin = 2
    for r in regions:
        y1, x1, y2, x2 = r.bbox
        b = [
            max(0, x1 - margin),
            max(0, y1 - margin),
            min(w, x2 + margin),
            min(h, y2 + margin)
        ]
        boxes.append(b)

    return depression_map, cand_mask, boxes


# ---------------------------------------------------------------------------
# Stage 3: Coordinate-Wise Bounding Box Jittering (CWBJ)
# ---------------------------------------------------------------------------

def jitter_boxes_cwbj(
    boxes_tensor: torch.Tensor,
    jitter_pixels: int = 3,
    max_w: int = 124,
    max_h: int = 124,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Applies Coordinate-Wise Bounding Box Jittering (CWBJ) as proposed in SinkSAM-Net.
    Each of the 4 corner coordinates (x1, y1, x2, y2) is perturbed independently.
    
    Args:
        boxes_tensor: Tensor of shape (K, 4) in [x1, y1, x2, y2]
        jitter_pixels: Maximum coordinate perturbation range [-jitter, +jitter]
        max_w: Maximum image width
        max_h: Maximum image height
    """
    if boxes_tensor.numel() == 0:
        return boxes_tensor
        
    dev = device or boxes_tensor.device
    perturbation = torch.randint(-jitter_pixels, jitter_pixels + 1, boxes_tensor.shape, device=dev).float()
    jittered = boxes_tensor + perturbation
    jittered[:, 0] = torch.clamp(jittered[:, 0], 0, max_w)
    jittered[:, 1] = torch.clamp(jittered[:, 1], 0, max_h)
    jittered[:, 2] = torch.clamp(jittered[:, 2], 0, max_w)
    jittered[:, 3] = torch.clamp(jittered[:, 3], 0, max_h)
    return jittered


# ---------------------------------------------------------------------------
# Stage 4: SAM Monte Carlo Inference
# ---------------------------------------------------------------------------

def run_sinksam_monte_carlo(
    predictor: Any,
    image_tensor: torch.Tensor,
    boxes: List[List[int]],
    num_samples: int = 20,
    jitter_pixels: int = 3,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes Monte Carlo SAM segmentation simulations using CWBJ bounding box prompts.
    
    Returns:
        mean_mask: Averaged foreground probability mask across N simulations (H, W)
        uncertainty_map: Epistemic Shannon entropy uncertainty map (H, W)
    """
    img_np = image_tensor.numpy() if isinstance(image_tensor, torch.Tensor) else image_tensor
    c, h, w = img_np.shape

    # Prepare false-color composite for SAM: NIR, Red, Green
    nir = img_np[3] if c > 3 else img_np[0]
    red = img_np[2] if c > 2 else img_np[1]
    green = img_np[1] if c > 1 else img_np[0]
    comp = np.stack([nir, red, green], axis=-1)
    comp_norm = ((comp - comp.min()) / (comp.max() - comp.min() + 1e-7) * 255.0).astype(np.uint8)

    # Set image into SAM predictor (computes ViT image embeddings once)
    predictor.set_image(comp_norm)

    if not boxes:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    boxes_tensor = torch.tensor(boxes, device=device, dtype=torch.float32)
    stacked_masks = []

    for _ in range(num_samples):
        j_boxes = jitter_boxes_cwbj(boxes_tensor, jitter_pixels=jitter_pixels, max_w=w, max_h=h, device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(j_boxes, (h, w))
        
        # Batch inference across all jittered prompt boxes
        with torch.no_grad():
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
        # Combine instance masks across boxes via element-wise maximum
        combined_mask, _ = torch.max(masks.squeeze(1).float(), dim=0)
        stacked_masks.append(combined_mask.cpu().numpy())

    stacked = np.stack(stacked_masks, axis=0)  # (N, H, W)
    mean_mask = np.mean(stacked, axis=0)

    # Epistemic Shannon uncertainty
    eps = 1e-7
    uncertainty_map = -mean_mask * np.log(mean_mask + eps) - (1.0 - mean_mask) * np.log(1.0 - mean_mask + eps)

    return mean_mask, uncertainty_map


# ---------------------------------------------------------------------------
# Stage 5: Shape-Aware Polsby-Popper Filtering
# ---------------------------------------------------------------------------

def compute_polsby_popper(region: Any) -> float:
    """
    Computes Polsby-Popper compactness score (Eq. 1 in Rafaeli et al., 2025):
    PP(R) = (4 * pi * Area) / (Perimeter^2)
    Scores range from 0 to 1, with higher values reflecting compact/circular shapes.
    """
    if region.perimeter == 0:
        return 1.0
    return float((4.0 * np.pi * region.area) / (region.perimeter ** 2))


def filter_pseudo_labels_shape(
    mean_mask: np.ndarray,
    depression_cand_mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 4,
    pp_min: float = 0.12
) -> np.ndarray:
    """
    Applies binarization and shape-aware geometric filtering using Polsby-Popper index.
    Restricts pseudo-labels to verified closed depression zones.
    """
    bin_mask = (mean_mask >= threshold).astype(np.uint8)
    labeled, num_features = label(bin_mask)
    regions = regionprops(labeled)

    filtered_mask = np.zeros_like(bin_mask, dtype=np.uint8)
    for r in regions:
        if r.area < min_area:
            continue
        pp = compute_polsby_popper(r)
        if pp >= pp_min:
            coords = r.coords
            filtered_mask[coords[:, 0], coords[:, 1]] = 1

    # Intersect with depression candidate mask to prevent bleeding into healthy forest
    refined_pseudo_label = filtered_mask * depression_cand_mask
    return refined_pseudo_label.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 6: SinkSAM Loss Function (Weighted Soft Dice + BCE)
# ---------------------------------------------------------------------------

class WeightedSoftDiceLoss(nn.Module):
    """
    Weighted Soft Dice Loss (WSDice) as defined in Eq. (2) of Rafaeli et al. (2025):
    L_WSDice = 1 - (2 * sum(w_s * y * y_hat) + 1) / (sum(w_s * y + w_n * (1 - y) * y_hat) + 1)
    
    where w_s = 1.0 (foreground sinkhole weight) and w_n = 0.07 (background weight).
    """
    def __init__(self, w_s: float = 1.0, w_n: float = 0.07, smooth: float = 1.0):
        super().__init__()
        self.w_s = w_s
        self.w_n = w_n
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        p_flat = probs.view(-1)
        t_flat = targets.view(-1)

        intersection = torch.sum(self.w_s * t_flat * p_flat)
        denominator = torch.sum(self.w_s * t_flat + self.w_n * (1.0 - t_flat) * p_flat)

        ws_dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - ws_dice


class SinkSAMLoss(nn.Module):
    """
    Full SinkSAM-Net loss combining Weighted Soft Dice (WSDice) and Binary Cross Entropy (BCE).
    """
    def __init__(self, w_s: float = 1.0, w_n: float = 0.07, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ws_dice = WeightedSoftDiceLoss(w_s=w_s, w_n=w_n)
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_dice = self.ws_dice(logits, targets)
        loss_bce = self.bce(logits, targets)
        return self.dice_weight * loss_dice + self.bce_weight * loss_bce


# ---------------------------------------------------------------------------
# Target Model: EffV2-UNet
# ---------------------------------------------------------------------------

def build_effv2_unet(in_channels: int = 9, encoder_name: str = "timm-efficientnet-b0") -> nn.Module:
    """
    Constructs the EffV2-UNet target model specified in Table 2 of Rafaeli et al. (2025).
    Wrapped with AutoPadModelWrapper to handle 124x124 spatial dimensions cleanly.
    """
    base_model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None  # Return raw logits
    )
    return AutoPadModelWrapper(base_model, multiple=32)
