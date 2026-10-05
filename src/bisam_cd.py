import os
import sys
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from skimage.segmentation import slic
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import BiTemporalCanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, stretch_rgb
from sam2.build_sam import build_sam2_video_predictor


class BiSAMCDPredictor:
    """
    BiSAM-CD: Zero-Shot Remote Sensing Change Detection via Bidirectional Temporal Memory in SAM2.
    Implements:
    1. BFSCAI: Bidirectional Frame Sequence Construction with Adaptive Interpolation.
    2. Memory-Augmented Tracking: Propagates object proposals through SAM2 video memory.
    3. TOTC-CSCV: Temporal Object Tracking Consistency & Cross-Sequence Change Verification.
    """
    def __init__(
        self,
        config_name: str = "configs/sam2/sam2_hiera_s.yaml",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        iou_threshold: float = 0.45,
        n_superpixels: int = 120
    ):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(MODELS_DIR, "sam2_hiera_small.pt")

        self.device = device
        self.iou_threshold = iou_threshold
        self.n_superpixels = n_superpixels

        print(f"Initializing BiSAM-CD with {config_name} and {checkpoint_path} on {device}...")
        self.predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=device)
        print("BiSAM-CD Video Predictor ready.")

    def _prepare_rgb(self, tensor_img: torch.Tensor) -> np.ndarray:
        """
        Converts PyTorch tensor (C, H, W) to uint8 RGB (H, W, 3).
        """
        arr = tensor_img.detach().cpu().numpy()
        if arr.shape[0] >= 3:
            # If multi-spectral, Red=2, Green=1, Blue=0
            rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
        else:
            rgb = np.repeat(arr[0:1], 3, axis=0).transpose(1, 2, 0)

        rgb = stretch_rgb(rgb)
        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return rgb_uint8

    def _compute_spectral_delta(self, t1_tensor: torch.Tensor, t2_tensor: torch.Tensor) -> np.ndarray:
        """
        Computes disturbance index magnitude (|T2 - T1|) from NBR or NDVI if available.
        """
        t1 = t1_tensor.detach().cpu().numpy()
        t2 = t2_tensor.detach().cpu().numpy()
        if t1.shape[0] >= 5:
            # Band 3=NIR, Band 4=SWIR1 (or Band 5=SWIR2)
            # NBR = (NIR - SWIR2) / (NIR + SWIR2)
            nir1, swir1 = t1[3], t1[4]
            nir2, swir2 = t2[3], t2[4]
            nbr1 = (nir1 - swir1) / (nir1 + swir1 + 1e-6)
            nbr2 = (nir2 - swir2) / (nir2 + swir2 + 1e-6)
            delta = np.maximum(0, nbr1 - nbr2)  # drop in NBR indicates canopy loss
            delta_norm = np.clip(delta / (np.percentile(delta, 98) + 1e-6), 0, 1)
            return delta_norm
        elif t1.shape[0] >= 4:
            nir1, red1 = t1[3], t1[2]
            nir2, red2 = t2[3], t2[2]
            ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-6)
            ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-6)
            delta = np.maximum(0, ndvi1 - ndvi2)
            delta_norm = np.clip(delta / (np.percentile(delta, 98) + 1e-6), 0, 1)
            return delta_norm
        else:
            diff = np.abs(t2 - t1).mean(axis=0)
            return np.clip(diff / (np.percentile(diff, 98) + 1e-6), 0, 1)

    def _generate_superpixel_prompts(self, rgb_img: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Generates object proposals via SLIC superpixels.
        """
        segments = slic(rgb_img, n_segments=self.n_superpixels, compactness=10, start_label=1)
        num_segments = segments.max()
        prompts = []

        for seg_val in range(1, num_segments + 1):
            mask = (segments == seg_val).astype(np.uint8)
            coords = np.argwhere(mask > 0)
            if len(coords) < 3:
                continue

            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            box = [float(xmin), float(ymin), float(xmax), float(ymax)]
            prompts.append({
                "obj_id": seg_val,
                "box": box,
                "mask": mask
            })

        return prompts, segments

    def _run_single_sequence(
        self,
        frame_list: List[np.ndarray],
        prompts: List[Dict],
        prompt_type: str = "box"
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Runs SAM2 video object tracking on a sequence of frames.
        """
        temp_dir = tempfile.mkdtemp(prefix="bisam_seq_")
        try:
            for i, frame in enumerate(frame_list):
                frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            inference_state = self.predictor.init_state(video_path=temp_dir)
            self.predictor.reset_state(inference_state)

            # Add prompts to frame 0
            for item in prompts:
                obj_id = item["obj_id"]
                if prompt_type == "box":
                    box_np = np.array(item["box"], dtype=np.float32)
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=box_np
                    )
                elif prompt_type == "mask":
                    mask_tensor = torch.from_numpy(item["mask"] > 0).to(torch.bool)
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        mask=mask_tensor
                    )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {}
                for idx, obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][obj_id] = (out_mask_logits[idx] > 0.0).cpu().numpy().squeeze()

            return video_segments
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def predict_change_map(
        self,
        t1_tensor: torch.Tensor,
        t2_tensor: torch.Tensor,
        prompt_type: str = "box"
    ) -> np.ndarray:
        """
        Bidirectional zero-shot inference:
        1. BFSCAI: Builds ascending (T1 -> Mid -> T2) and descending (T2 -> Mid -> T1) sequences.
        2. Memory tracking: tracks object proposals across both sequences.
        3. TOTC-CSCV: Temporal object tracking consistency & cross-sequence validation.
        4. Fuses tracking inconsistency with spectral disturbance magnitude.
        """
        rgb1 = self._prepare_rgb(t1_tensor)
        rgb2 = self._prepare_rgb(t2_tensor)
        h, w = rgb1.shape[:2]

        # BFSCAI: Adaptive intermediate interpolated frame
        rgb_mid = ((0.5 * rgb1.astype(np.float32) + 0.5 * rgb2.astype(np.float32))).astype(np.uint8)

        seq_asc = [rgb1, rgb_mid, rgb2]
        seq_desc = [rgb2, rgb_mid, rgb1]

        prompts_1, segs1 = self._generate_superpixel_prompts(rgb1)
        prompts_2, segs2 = self._generate_superpixel_prompts(rgb2)

        # 1. Forward tracking (T1 -> T2)
        out_asc = self._run_single_sequence(seq_asc, prompts_1, prompt_type=prompt_type)
        # 2. Backward tracking (T2 -> T1)
        out_desc = self._run_single_sequence(seq_desc, prompts_2, prompt_type=prompt_type)

        tracking_anomaly_asc = np.zeros((h, w), dtype=np.float32)
        if 0 in out_asc and 2 in out_asc:
            for obj_id, m_start in out_asc[0].items():
                m_end = out_asc[2].get(obj_id, np.zeros_like(m_start))
                inter = np.logical_and(m_start > 0, m_end > 0).sum()
                union = np.logical_or(m_start > 0, m_end > 0).sum()
                iou = inter / (union + 1e-6)
                if iou < self.iou_threshold:
                    # Inconsistency score = 1 - IoU
                    inconsistency = 1.0 - iou
                    tracking_anomaly_asc = np.maximum(tracking_anomaly_asc, (m_start > 0).astype(np.float32) * inconsistency)

        tracking_anomaly_desc = np.zeros((h, w), dtype=np.float32)
        if 0 in out_desc and 2 in out_desc:
            for obj_id, m_start in out_desc[0].items():
                m_end = out_desc[2].get(obj_id, np.zeros_like(m_start))
                inter = np.logical_and(m_start > 0, m_end > 0).sum()
                union = np.logical_or(m_start > 0, m_end > 0).sum()
                iou = inter / (union + 1e-6)
                if iou < self.iou_threshold:
                    inconsistency = 1.0 - iou
                    tracking_anomaly_desc = np.maximum(tracking_anomaly_desc, (m_start > 0).astype(np.float32) * inconsistency)

        # Cross-sequence consensus
        bidirectional_tracking_score = 0.5 * tracking_anomaly_asc + 0.5 * tracking_anomaly_desc

        # Physical spectral disturbance index
        spectral_delta = self._compute_spectral_delta(t1_tensor, t2_tensor)

        # Synergistic fusion: tracking discrepancy amplified by spectral canopy gap signal
        combined_prob = 0.6 * bidirectional_tracking_score + 0.4 * spectral_delta

        # Morphological opening to suppress single-pixel impulse noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        filtered_prob = cv2.morphologyEx(combined_prob.astype(np.float32), cv2.MORPH_OPEN, kernel)
        return filtered_prob


def evaluate_bisam_cd(
    predictor: BiSAMCDPredictor,
    val_loader: DataLoader,
    max_samples: Optional[int] = None,
    threshold: float = 0.45
) -> Dict[str, float]:
    """
    Evaluates BiSAM-CD in zero-shot mode across the validation split.
    """
    all_preds = []
    all_targets = []

    count = 0
    for t1, t2, gt_mask, _ in val_loader:
        pred_map = predictor.predict_change_map(t1[0], t2[0], prompt_type="box")
        gt = gt_mask[0, 0].cpu().numpy()

        all_preds.append(pred_map.flatten())
        all_targets.append(gt.flatten())

        count += 1
        if max_samples and count >= max_samples:
            break

    preds_cat = np.concatenate(all_preds)
    targets_cat = np.concatenate(all_targets)
    metrics = compute_binary_metrics(preds_cat, targets_cat, threshold=threshold)
    return metrics
