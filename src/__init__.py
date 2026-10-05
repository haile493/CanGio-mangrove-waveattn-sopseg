"""
WaveAttn SOPSeg: High-Precision Mangrove Canopy Disturbance Mapping via Frequency-Aware Prompted Segmentation.
Can Gio Mangrove Biosphere Reserve Satellite Image Segmentation Package.
"""

__version__ = "1.0.0"

from src.models import build_segmentation_model
from src.sam_satellite import LoRALinear, apply_lora_to_sam, ensure_sam_checkpoint
from src.sopseg import (
    RegionAdaptiveMagnification,
    OrientedPromptMechanism,
    SOPSegMaskDecoder,
    SOPSegMultiTaskLoss
)
from src.sopseg_enhanced import (
    EnhancedSOPSegModel,
    WaveAttnSOPSegModel,
    HaarDWT2D,
    HaarIDWT2D,
    WaveFusionRefinementBlock,
    GLCAMContextAttention,
    EnhancedProgressiveRefinementModule
)
from src.dataset import CanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, compute_boundary_iou

__all__ = [
    "WaveAttnSOPSegModel",
    "EnhancedSOPSegModel",
    "build_segmentation_model",
    "LoRALinear",
    "apply_lora_to_sam",
    "ensure_sam_checkpoint",
    "RegionAdaptiveMagnification",
    "OrientedPromptMechanism",
    "SOPSegMaskDecoder",
    "SOPSegMultiTaskLoss",
    "HaarDWT2D",
    "HaarIDWT2D",
    "WaveFusionRefinementBlock",
    "GLCAMContextAttention",
    "EnhancedProgressiveRefinementModule",
    "CanGioDataset",
    "discover_scenes",
    "compute_binary_metrics",
    "compute_boundary_iou"
]
