"""
Unified Experiment Runner for SOPSeg Enhanced (Option 1) and Bi-Temporal SOPSeg-CD (Option 2).
Integrating SAM2-CD (WaveFusion, GLCAM, ASG) and BiSAM-CD (Bi-temporal tracking, NBR guidance).

Usage:
    python scripts/run_sopseg_integrated_cd.py --option 1 --source both --epochs 12
    python scripts/run_sopseg_integrated_cd.py --option 2 --source both --epochs 12
    python scripts/run_sopseg_integrated_cd.py --all --epochs 12
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    Config,
    DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    STUDY_AREA_BOUNDS
)
from src.dataset import (
    CanGioDataset,
    BiTemporalCanGioDataset,
    discover_scenes
)
from src.evaluate import (
    compute_binary_metrics,
    compute_boundary_iou,
    stretch_rgb
)
from src.sopseg import (
    RegionAdaptiveMagnification,
    OrientedPromptMechanism,
    apply_position_embedding_interpolation,
    SOPSegMultiTaskLoss
)
from src.sopseg_enhanced import (
    EnhancedSOPSegModel
)
from src.sopseg_bitemporal import (
    BiTemporalSOPSegModel
)
from segment_anything import sam_model_registry


# ---------------------------------------------------------------------------
# 1. Dataset Extraction: Option 1 (Single-Date with NBR Guidance)
# ---------------------------------------------------------------------------

def extract_enhanced_instances(
    scene_image: np.ndarray,
    scene_gt: np.ndarray,
    scene_meta: Dict[str, Any],
    min_instance_area: int = 1
) -> List[Dict[str, Any]]:
    """
    Extracts connected disturbance components with:
    - 3-channel composite [NIR, Red, Green] for SAM ViT
    - 4-channel guidance [NIR, Red, Green, NBR] for WaveFusion PRM
    """
    bin_mask = (scene_gt.squeeze() > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)

    instances = []
    c, h, w = scene_image.shape

    nir = scene_image[3] if c > 3 else scene_image[2]
    red = scene_image[2]
    green = scene_image[1]
    composite = np.stack([nir, red, green], axis=0)  # (3, H, W)
    c_min = composite.min(axis=(1, 2), keepdims=True)
    c_max = composite.max(axis=(1, 2), keepdims=True)
    comp_norm = (composite - c_min) / (c_max - c_min + 1e-7) * 255.0

    # Biophysical NBR canopy moisture/gap index
    if c >= 5:
        swir2 = scene_image[5] if c > 5 else scene_image[4]
        nbr = (nir - swir2) / (nir + swir2 + 1e-6)
    else:
        nbr = (nir - red) / (nir + red + 1e-6)
    nbr_norm = ((nbr - nbr.min()) / (nbr.max() - nbr.min() + 1e-7) * 255.0).astype(np.float32)

    # 4-channel guidance: [NIR, Red, Green, NBR]
    guidance = np.concatenate([comp_norm, nbr_norm[np.newaxis, :, :]], axis=0)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_instance_area:
            continue

        bx = stats[lbl, cv2.CC_STAT_LEFT]
        by = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        inst_mask = (labels == lbl).astype(np.uint8)

        instances.append({
            "image": comp_norm,
            "guidance": guidance,
            "scene_gt": bin_mask,
            "instance_mask": inst_mask,
            "bbox": (bx, by, bw, bh),
            "area": area,
            "dim": max(bw, bh),
            "meta": scene_meta
        })

    return instances


class EnhancedSOPSegDataset(Dataset):
    """Dataset for Option 1: Enhanced SOPSeg (WaveFusion + GLCAM + NBR)."""
    def __init__(
        self,
        instance_records: List[Dict[str, Any]],
        ram: RegionAdaptiveMagnification,
        is_training: bool = True
    ):
        self.records = instance_records
        self.ram = ram
        self.is_training = is_training
        self.opm = OrientedPromptMechanism()

        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img = rec["image"]
        guide = rec["guidance"]
        mask = rec["instance_mask"]
        bbox = rec["bbox"]

        # RAM extraction
        ram_res = self.ram.extract_region(img, bbox, mask=mask, is_training=self.is_training)
        patch_img = ram_res["patch_img"].astype(np.float32)
        patch_mask = ram_res["patch_mask"].astype(np.float32)

        # Extract matching guidance patch
        ram_guide = self.ram.extract_region(guide, bbox, mask=None, is_training=self.is_training)
        patch_guide = ram_guide["patch_img"].astype(np.float32)

        # Augmentation
        if self.is_training:
            if np.random.rand() > 0.5:
                patch_img = np.flip(patch_img, axis=2).copy()
                patch_guide = np.flip(patch_guide, axis=2).copy()
                patch_mask = np.fliplr(patch_mask).copy()
            if np.random.rand() > 0.5:
                patch_img = np.flip(patch_img, axis=1).copy()
                patch_guide = np.flip(patch_guide, axis=1).copy()
                patch_mask = np.flipud(patch_mask).copy()

        pts, lbls, box = self.opm.extract_oriented_prompts(patch_mask)

        # Normalize 3-channel image for SAM ViT
        norm_img = (patch_img - self.pixel_mean) / self.pixel_std
        # Normalize 4-channel guidance to [0, 1]
        norm_guide = patch_guide / 255.0

        return {
            "image": torch.from_numpy(norm_img).float(),
            "guidance": torch.from_numpy(norm_guide).float(),
            "mask": torch.from_numpy(patch_mask).unsqueeze(0).float(),
            "points": torch.from_numpy(pts).float(),
            "point_labels": torch.from_numpy(lbls).int(),
            "boxes": torch.from_numpy(box).float(),
            "meta": rec["meta"],
            "bbox": bbox,
            "ram_meta": ram_res["meta"],
            "area": rec["area"],
            "dim": rec["dim"]
        }


# ---------------------------------------------------------------------------
# 2. Dataset Extraction: Option 2 (Bi-Temporal Change Detection)
# ---------------------------------------------------------------------------

def extract_bitemporal_instances(
    img1: np.ndarray,
    img2: np.ndarray,
    change_gt: np.ndarray,
    scene_meta: Dict[str, Any],
    min_instance_area: int = 1
) -> List[Dict[str, Any]]:
    """
    Extracts synchronized disturbance change components between T1 and T2:
    - 3-channel composite T1 and 3-channel composite T2
    - 7-channel bi-temporal guidance [Comp1, Comp2, delta_NBR]
    """
    bin_mask = (change_gt.squeeze() > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)

    instances = []
    c, h, w = img1.shape

    # T1 composite
    nir1 = img1[3] if c > 3 else img1[2]
    red1 = img1[2]
    green1 = img1[1]
    comp1 = np.stack([nir1, red1, green1], axis=0)
    c1_min = comp1.min(axis=(1, 2), keepdims=True)
    c1_max = comp1.max(axis=(1, 2), keepdims=True)
    comp1_norm = (comp1 - c1_min) / (c1_max - c1_min + 1e-7) * 255.0

    # T2 composite
    nir2 = img2[3] if c > 3 else img2[2]
    red2 = img2[2]
    green2 = img2[1]
    comp2 = np.stack([nir2, red2, green2], axis=0)
    c2_min = comp2.min(axis=(1, 2), keepdims=True)
    c2_max = comp2.max(axis=(1, 2), keepdims=True)
    comp2_norm = (comp2 - c2_min) / (c2_max - c2_min + 1e-7) * 255.0

    # Biophysical delta NBR
    if c >= 5:
        swir2_1 = img1[5] if c > 5 else img1[4]
        swir2_2 = img2[5] if c > 5 else img2[4]
        nbr1 = (nir1 - swir2_1) / (nir1 + swir2_1 + 1e-6)
        nbr2 = (nir2 - swir2_2) / (nir2 + swir2_2 + 1e-6)
    else:
        nbr1 = (nir1 - red1) / (nir1 + red1 + 1e-6)
        nbr2 = (nir2 - red2) / (nir2 + red2 + 1e-6)

    delta_nbr = np.maximum(0, nbr1 - nbr2)  # canopy loss
    delta_norm = ((delta_nbr - delta_nbr.min()) / (delta_nbr.max() - delta_nbr.min() + 1e-7) * 255.0).astype(np.float32)

    # 7-channel guidance: [comp1 (3), comp2 (3), delta_nbr (1)]
    bitemp_guide = np.concatenate([comp1_norm, comp2_norm, delta_norm[np.newaxis, :, :]], axis=0)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_instance_area:
            continue

        bx = stats[lbl, cv2.CC_STAT_LEFT]
        by = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        inst_mask = (labels == lbl).astype(np.uint8)

        instances.append({
            "comp1": comp1_norm,
            "comp2": comp2_norm,
            "guidance": bitemp_guide,
            "change_gt": bin_mask,
            "instance_mask": inst_mask,
            "bbox": (bx, by, bw, bh),
            "area": area,
            "dim": max(bw, bh),
            "meta": scene_meta
        })

    return instances


class BiTemporalSOPSegDataset(Dataset):
    """Dataset for Option 2: Bi-Temporal Siamese SOPSeg-CD."""
    def __init__(
        self,
        instance_records: List[Dict[str, Any]],
        ram: RegionAdaptiveMagnification,
        is_training: bool = True
    ):
        self.records = instance_records
        self.ram = ram
        self.is_training = is_training
        self.opm = OrientedPromptMechanism()

        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        comp1 = rec["comp1"]
        comp2 = rec["comp2"]
        guide = rec["guidance"]
        mask = rec["instance_mask"]
        bbox = rec["bbox"]

        # Synchronized RAM extraction across both timestamps
        ram_res1 = self.ram.extract_region(comp1, bbox, mask=mask, is_training=self.is_training)
        patch_t1 = ram_res1["patch_img"].astype(np.float32)
        patch_mask = ram_res1["patch_mask"].astype(np.float32)

        ram_res2 = self.ram.extract_region(comp2, bbox, mask=None, is_training=self.is_training)
        patch_t2 = ram_res2["patch_img"].astype(np.float32)

        ram_guide = self.ram.extract_region(guide, bbox, mask=None, is_training=self.is_training)
        patch_guide = ram_guide["patch_img"].astype(np.float32)

        if self.is_training:
            if np.random.rand() > 0.5:
                patch_t1 = np.flip(patch_t1, axis=2).copy()
                patch_t2 = np.flip(patch_t2, axis=2).copy()
                patch_guide = np.flip(patch_guide, axis=2).copy()
                patch_mask = np.fliplr(patch_mask).copy()
            if np.random.rand() > 0.5:
                patch_t1 = np.flip(patch_t1, axis=1).copy()
                patch_t2 = np.flip(patch_t2, axis=1).copy()
                patch_guide = np.flip(patch_guide, axis=1).copy()
                patch_mask = np.flipud(patch_mask).copy()

        pts, lbls, box = self.opm.extract_oriented_prompts(patch_mask)

        norm_t1 = (patch_t1 - self.pixel_mean) / self.pixel_std
        norm_t2 = (patch_t2 - self.pixel_mean) / self.pixel_std
        norm_guide = patch_guide / 255.0

        return {
            "patch_t1": torch.from_numpy(norm_t1).float(),
            "patch_t2": torch.from_numpy(norm_t2).float(),
            "guidance": torch.from_numpy(norm_guide).float(),
            "mask": torch.from_numpy(patch_mask).unsqueeze(0).float(),
            "points": torch.from_numpy(pts).float(),
            "point_labels": torch.from_numpy(lbls).int(),
            "boxes": torch.from_numpy(box).float(),
            "meta": rec["meta"],
            "bbox": bbox,
            "ram_meta": ram_res1["meta"],
            "area": rec["area"],
            "dim": rec["dim"]
        }


# ---------------------------------------------------------------------------
# 3. Model Builder Helpers
# ---------------------------------------------------------------------------

def build_option1_model(
    sam_checkpoint: str,
    device: str = "cuda"
) -> EnhancedSOPSegModel:
    """Builds Option 1: Enhanced SOPSeg with WaveFusion & GLCAM + LoRA."""
    base_sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    apply_position_embedding_interpolation(base_sam, target_size=256)
    model = EnhancedSOPSegModel(
        base_sam=base_sam,
        shallow_layer_idx=3,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16.0,
        guidance_channels=4
    )
    return model.to(device)


def build_option2_model(
    sam_checkpoint: str,
    device: str = "cuda"
) -> BiTemporalSOPSegModel:
    """Builds Option 2: Bi-Temporal Siamese SOPSeg-CD + LoRA."""
    base_sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    apply_position_embedding_interpolation(base_sam, target_size=256)
    model = BiTemporalSOPSegModel(
        base_sam=base_sam,
        shallow_layer_idx=3,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16.0
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# 4. Training & Evaluation Engine
# ---------------------------------------------------------------------------

def train_and_eval_option1(
    source: str = "anhgoc",
    epochs: int = 12,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda"
) -> Tuple[EnhancedSOPSegModel, Dict[str, Any]]:
    print(f"\n=======================================================")
    print(f" OPTION 1: Enhanced SOPSeg (WaveFusion + GLCAM + NBR) on {source.upper()}")
    print(f"=======================================================")

    sam_ckpt = os.path.join(MODELS_DIR, "sam_vit_b_01ec64.pth")
    if not os.path.exists(sam_ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found at {sam_ckpt}")

    model = build_option1_model(sam_ckpt, device=device)
    loss_fn = SOPSegMultiTaskLoss(lambda_iou=5.0)
    ram = RegionAdaptiveMagnification(s_in=256)

    # Collect scenes
    scenes = discover_scenes(DATA_DIR, source=source)
    val_years = ["2011", "2014"]
    train_scenes = [s for s in scenes if not any(y in str(s.get("date", "")) for y in val_years)]
    val_scenes = [s for s in scenes if any(y in str(s.get("date", "")) for y in val_years)]

    dataset_cls = CanGioDataset(scenes=scenes, require_gt=True)
    train_instances = []
    for sc in train_scenes:
        img, gt, meta = dataset_cls.load_scene_numpy(sc)
        train_instances.extend(extract_enhanced_instances(img, gt, meta, min_instance_area=1))

    val_instances = []
    val_scene_records = []
    for sc in val_scenes:
        img, gt, meta = dataset_cls.load_scene_numpy(sc)
        insts = extract_enhanced_instances(img, gt, meta, min_instance_area=1)
        val_instances.extend(insts)
        val_scene_records.append((img, gt, meta, insts))

    print(f"[{source.upper()}] Train instances: {len(train_instances)} | Val instances: {len(val_instances)}")

    train_ds = EnhancedSOPSegDataset(train_instances, ram, is_training=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_iou = 0.0
    best_metrics = {}
    best_ckpt_path = os.path.join(MODELS_DIR, f"best_sopseg_enhanced_{source}.pth")

    for ep in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            guides = batch["guidance"].to(device)
            masks = batch["mask"].to(device)
            points = batch["points"].to(device)
            p_labels = batch["point_labels"].to(device)
            boxes = batch["boxes"].to(device)

            optimizer.zero_grad()
            preds = model(
                image_patch=imgs,
                guidance_patch=guides,
                points=(points, p_labels),
                boxes=boxes
            )

            losses = loss_fn(preds, masks)
            loss = losses["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_epoch_loss += loss.item()

        scheduler.step()
        avg_loss = total_epoch_loss / max(1, len(train_loader))

        # Evaluate at epoch end
        val_eval = evaluate_option1(model, val_scene_records, ram, device=device)
        curr_iou = val_eval["global_metrics"]["iou"]
        print(f"Epoch {ep:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Val IoU: {curr_iou:.4f} | BIoU: {val_eval['global_metrics']['biou']:.4f}", flush=True)

        if curr_iou > best_iou:
            best_iou = curr_iou
            best_metrics = val_eval
            torch.save(model.state_dict(), best_ckpt_path)

    print(f"\n[Done Option 1 on {source.upper()}] Best Val IoU = {best_iou:.4f} saved to {best_ckpt_path}")
    return model, best_metrics


def evaluate_option1(
    model: EnhancedSOPSegModel,
    val_scene_records: List[Tuple],
    ram: RegionAdaptiveMagnification,
    device: str = "cuda"
) -> Dict[str, Any]:
    model.eval()
    all_preds_flat = []
    all_gts_flat = []
    pred_scenes = []

    pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
    opm = OrientedPromptMechanism()

    with torch.no_grad():
        for img, gt, meta, instances in val_scene_records:
            h, w = gt.shape[-2:]
            scene_pred = np.zeros((h, w), dtype=np.float32)

            for inst in instances:
                bbox = inst["bbox"]
                inst_mask = inst["instance_mask"]
                ram_res = ram.extract_region(inst["image"], bbox, mask=inst_mask, is_training=False)
                ram_guide = ram.extract_region(inst["guidance"], bbox, mask=None, is_training=False)

                patch_img = ram_res["patch_img"].astype(np.float32)
                patch_mask = ram_res["patch_mask"].astype(np.float32)
                patch_guide = ram_guide["patch_img"].astype(np.float32)

                pts, lbls, box = opm.extract_oriented_prompts(patch_mask)

                norm_img = (patch_img - pixel_mean) / pixel_std
                norm_guide = patch_guide / 255.0

                t_img = torch.from_numpy(norm_img).unsqueeze(0).float().to(device)
                t_guide = torch.from_numpy(norm_guide).unsqueeze(0).float().to(device)
                t_pts = torch.from_numpy(pts).unsqueeze(0).float().to(device)
                t_lbls = torch.from_numpy(lbls).unsqueeze(0).int().to(device)
                t_box = torch.from_numpy(box).unsqueeze(0).float().to(device)

                preds = model(
                    image_patch=t_img,
                    guidance_patch=t_guide,
                    points=(t_pts, t_lbls),
                    boxes=t_box
                )

                prob_patch = torch.sigmoid(preds["p1"][:, 0:1, :, :]).squeeze().cpu().numpy()
                scene_pred = ram.inverse_map_patch_to_canvas(prob_patch, ram_res["meta"], scene_pred)

            pred_bin = (scene_pred >= 0.5).astype(np.uint8)
            all_preds_flat.append(pred_bin.flatten())
            all_gts_flat.append(gt.flatten())
            pred_scenes.append((scene_pred, pred_bin, gt, meta))

    preds_cat = np.concatenate(all_preds_flat)
    gts_cat = np.concatenate(all_gts_flat)
    global_metrics = compute_binary_metrics(preds_cat, gts_cat, threshold=0.5)

    # Compute Boundary IoU across validation scenes
    bious = []
    for _, p_bin, gt_arr, _ in pred_scenes:
        b_iou = compute_boundary_iou(p_bin, gt_arr.squeeze(), dilation_ratio=0.02)
        bious.append(b_iou)
    global_metrics["biou"] = float(np.mean(bious))

    return {
        "global_metrics": global_metrics,
        "pred_scenes": pred_scenes
    }


def train_and_eval_option2(
    source: str = "anhgoc",
    epochs: int = 12,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda"
) -> Tuple[BiTemporalSOPSegModel, Dict[str, Any]]:
    print(f"\n=======================================================")
    print(f" OPTION 2: Bi-Temporal Siamese SOPSeg-CD on {source.upper()}")
    print(f"=======================================================")

    sam_ckpt = os.path.join(MODELS_DIR, "sam_vit_b_01ec64.pth")
    model = build_option2_model(sam_ckpt, device=device)
    loss_fn = SOPSegMultiTaskLoss(lambda_iou=5.0)
    ram = RegionAdaptiveMagnification(s_in=256)

    scenes = discover_scenes(DATA_DIR, source=source)
    bids = BiTemporalCanGioDataset(scenes=scenes)
    val_years = ["2011", "2014"]

    train_instances = []
    val_instances = []
    val_pair_records = []

    for idx1, idx2 in bids.pairs:
        sc1 = bids.base_ds.scenes[idx1]
        sc2 = bids.base_ds.scenes[idx2]
        img1, _, meta1 = bids.base_ds.load_scene_numpy(sc1)
        img2, mask2, meta2 = bids.base_ds.load_scene_numpy(sc2)

        meta = {
            "t1_scene": meta1["scene_id"],
            "t2_scene": meta2["scene_id"],
            "t2_date": meta2["date"],
            "crs": meta2["crs"],
            "transform": meta2["transform"]
        }

        insts = extract_bitemporal_instances(img1, img2, mask2, meta, min_instance_area=1)
        is_val = any(y in str(sc2.get("date", "")) for y in val_years)

        if is_val:
            val_instances.extend(insts)
            val_pair_records.append((img1, img2, mask2, meta, insts))
        else:
            train_instances.extend(insts)

    print(f"[{source.upper()}-CD] Train instances: {len(train_instances)} | Val instances: {len(val_instances)}")

    train_ds = BiTemporalSOPSegDataset(train_instances, ram, is_training=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_iou = 0.0
    best_metrics = {}
    best_ckpt_path = os.path.join(MODELS_DIR, f"best_sopseg_bitemporal_{source}.pth")

    for ep in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0

        for batch in train_loader:
            t1 = batch["patch_t1"].to(device)
            t2 = batch["patch_t2"].to(device)
            guides = batch["guidance"].to(device)
            masks = batch["mask"].to(device)
            points = batch["points"].to(device)
            p_labels = batch["point_labels"].to(device)
            boxes = batch["boxes"].to(device)

            optimizer.zero_grad()
            preds = model(
                patch_t1=t1,
                patch_t2=t2,
                bitemp_guidance=guides,
                points=(points, p_labels),
                boxes=boxes
            )

            losses = loss_fn(preds, masks)
            loss = losses["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_epoch_loss += loss.item()

        scheduler.step()
        avg_loss = total_epoch_loss / max(1, len(train_loader))

        val_eval = evaluate_option2(model, val_pair_records, ram, device=device)
        curr_iou = val_eval["global_metrics"]["iou"]
        print(f"Epoch {ep:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Val IoU: {curr_iou:.4f} | BIoU: {val_eval['global_metrics']['biou']:.4f}", flush=True)

        if curr_iou > best_iou:
            best_iou = curr_iou
            best_metrics = val_eval
            torch.save(model.state_dict(), best_ckpt_path)

    print(f"\n[Done Option 2 on {source.upper()}] Best Val IoU = {best_iou:.4f} saved to {best_ckpt_path}")
    return model, best_metrics


def evaluate_option2(
    model: BiTemporalSOPSegModel,
    val_pair_records: List[Tuple],
    ram: RegionAdaptiveMagnification,
    device: str = "cuda"
) -> Dict[str, Any]:
    model.eval()
    all_preds_flat = []
    all_gts_flat = []
    pred_scenes = []

    pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
    opm = OrientedPromptMechanism()

    with torch.no_grad():
        for img1, img2, change_gt, meta, instances in val_pair_records:
            h, w = change_gt.shape[-2:]
            scene_pred = np.zeros((h, w), dtype=np.float32)

            for inst in instances:
                bbox = inst["bbox"]
                inst_mask = inst["instance_mask"]
                ram_res1 = ram.extract_region(inst["comp1"], bbox, mask=inst_mask, is_training=False)
                ram_res2 = ram.extract_region(inst["comp2"], bbox, mask=None, is_training=False)
                ram_guide = ram.extract_region(inst["guidance"], bbox, mask=None, is_training=False)

                patch_t1 = ram_res1["patch_img"].astype(np.float32)
                patch_t2 = ram_res2["patch_img"].astype(np.float32)
                patch_mask = ram_res1["patch_mask"].astype(np.float32)
                patch_guide = ram_guide["patch_img"].astype(np.float32)

                pts, lbls, box = opm.extract_oriented_prompts(patch_mask)

                norm_t1 = (patch_t1 - pixel_mean) / pixel_std
                norm_t2 = (patch_t2 - pixel_mean) / pixel_std
                norm_guide = patch_guide / 255.0

                t_t1 = torch.from_numpy(norm_t1).unsqueeze(0).float().to(device)
                t_t2 = torch.from_numpy(norm_t2).unsqueeze(0).float().to(device)
                t_guide = torch.from_numpy(norm_guide).unsqueeze(0).float().to(device)
                t_pts = torch.from_numpy(pts).unsqueeze(0).float().to(device)
                t_lbls = torch.from_numpy(lbls).unsqueeze(0).int().to(device)
                t_box = torch.from_numpy(box).unsqueeze(0).float().to(device)

                preds = model(
                    patch_t1=t_t1,
                    patch_t2=t_t2,
                    bitemp_guidance=t_guide,
                    points=(t_pts, t_lbls),
                    boxes=t_box
                )

                prob_patch = torch.sigmoid(preds["p1"][:, 0:1, :, :]).squeeze().cpu().numpy()
                scene_pred = ram.inverse_map_patch_to_canvas(prob_patch, ram_res1["meta"], scene_pred)

            pred_bin = (scene_pred >= 0.5).astype(np.uint8)
            all_preds_flat.append(pred_bin.flatten())
            all_gts_flat.append(change_gt.flatten())
            pred_scenes.append((scene_pred, pred_bin, change_gt, meta))

    preds_cat = np.concatenate(all_preds_flat)
    gts_cat = np.concatenate(all_gts_flat)
    global_metrics = compute_binary_metrics(preds_cat, gts_cat, threshold=0.5)

    bious = []
    for _, p_bin, gt_arr, _ in pred_scenes:
        b_iou = compute_boundary_iou(p_bin, gt_arr.squeeze(), dilation_ratio=0.02)
        bious.append(b_iou)
    global_metrics["biou"] = float(np.mean(bious))

    return {
        "global_metrics": global_metrics,
        "pred_scenes": pred_scenes
    }


# ---------------------------------------------------------------------------
# 5. Export GeoTIFF Predictions
# ---------------------------------------------------------------------------

def export_geotiffs(pred_scenes: List[Tuple], prefix: str):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    for scene_prob, pred_bin, gt, meta in pred_scenes:
        scene_id = meta.get("scene_id", meta.get("t2_scene", "unknown"))
        out_tif = os.path.join(PREDICTIONS_DIR, f"{prefix}_{scene_id}.tif")

        crs = meta.get("crs", "EPSG:32648")
        trans = meta.get("transform")
        h, w = pred_bin.shape

        with rasterio.open(
            out_tif,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=2,
            dtype=rasterio.float32,
            crs=crs,
            transform=trans
        ) as dst:
            dst.write(pred_bin.astype(np.float32), 1)
            dst.write(scene_prob.astype(np.float32), 2)
            dst.set_band_description(1, "BinaryDisturbanceMask")
            dst.set_band_description(2, "DisturbanceProbability")

        print(f"  [GeoTIFF Exported] {out_tif}")


# ---------------------------------------------------------------------------
# 6. Master Benchmark Compilation & Figure Generation
# ---------------------------------------------------------------------------

def compile_benchmark_and_plots(
    opt1_cut_metrics: Dict[str, Any],
    opt1_anhgoc_metrics: Dict[str, Any],
    opt2_cut_metrics: Dict[str, Any],
    opt2_anhgoc_metrics: Dict[str, Any]
):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_json_path = os.path.join(RESULTS_DIR, "sopseg_integrated_benchmark_results.json")
    results = {
        "option1_enhanced_sopseg": {
            "cut": opt1_cut_metrics["global_metrics"],
            "anhgoc": opt1_anhgoc_metrics["global_metrics"]
        },
        "option2_bitemporal_sopseg": {
            "cut": opt2_cut_metrics["global_metrics"],
            "anhgoc": opt2_anhgoc_metrics["global_metrics"]
        }
    }

    with open(summary_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] Master benchmark saved to: {summary_json_path}")

    # 1. Figure: Option 1 vs Option 2 across Cut & Anhgoc
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    metrics = ["iou", "biou", "dice", "precision"]
    titles = ["IoU", "Boundary IoU (BIoU)", "Dice (F1)", "Precision"]

    opt1_c = [opt1_cut_metrics["global_metrics"][m] for m in metrics]
    opt1_a = [opt1_anhgoc_metrics["global_metrics"][m] for m in metrics]
    opt2_c = [opt2_cut_metrics["global_metrics"][m] for m in metrics]
    opt2_a = [opt2_anhgoc_metrics["global_metrics"][m] for m in metrics]

    methods = [
        "Opt 1 (WaveAttn Cut)",
        "Opt 1 (WaveAttn Anhgoc)",
        "Opt 2 (SiameseCD Cut)",
        "Opt 2 (SiameseCD Anhgoc)"
    ]

    for idx, (m_key, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        vals = [opt1_c[idx], opt1_a[idx], opt2_c[idx], opt2_a[idx]]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        bars = ax.bar(methods, vals, color=colors, alpha=0.88, edgecolor="black", width=0.55)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_xticklabels(methods, rotation=18, ha="right", fontsize=8.5, fontweight="bold")

        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("SAM2-CD & BiSAM-CD Integration: Option 1 (WaveAttn) vs. Option 2 (Bi-Temporal Siamese CD)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_p1 = os.path.join(PLOTS_DIR, "sopseg_option1_vs_option2_benchmark.png")
    plt.savefig(plot_p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure Saved] {plot_p1}")

    # 2. Figure: Master Project Comparison (All 7 Paradigms)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    all_methods = [
        "U-Net",
        "WSL (Study 1)",
        "SAM + LoRA",
        "SOPSeg (Frozen)",
        "SOPSeg + LoRA",
        "Opt 1 (WaveAttn)",
        "Opt 2 (SiameseCD)"
    ]
    # Anhgoc values across paradigms
    iou_all = [0.3782, 0.0147, 0.5413, 0.7378, 0.7474, opt1_anhgoc_metrics["global_metrics"]["iou"], opt2_anhgoc_metrics["global_metrics"]["iou"]]
    biou_all = [0.2740, 0.0090, 0.4180, 0.9641, 0.9654, opt1_anhgoc_metrics["global_metrics"]["biou"], opt2_anhgoc_metrics["global_metrics"]["biou"]]
    dice_all = [0.5488, 0.0290, 0.7024, 0.8492, 0.8554, opt1_anhgoc_metrics["global_metrics"]["dice"], opt2_anhgoc_metrics["global_metrics"]["dice"]]

    for ax, data, m_title in zip(axes, [iou_all, biou_all, dice_all], ["IoU", "Boundary IoU (BIoU)", "Dice (F1)"]):
        bars = ax.bar(all_methods, data, color=["#7f7f7f", "#8c564b", "#9467bd", "#1f77b4", "#e377c2", "#ff7f0e", "#2ca02c"], edgecolor="black", width=0.6)
        ax.set_title(m_title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_xticklabels(all_methods, rotation=22, ha="right", fontsize=8.5, fontweight="bold")
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Can Gio Mangrove Benchmark: Master Paradigm Comparison Across 7 Methods", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_master = os.path.join(PLOTS_DIR, "sopseg_integrated_master_comparison.png")
    plt.savefig(plot_master, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure Saved] {plot_master}")

    # Copy to brain artifact directory
    brain_dir = os.path.join("C:\\Users\\hai49\\.gemini\\antigravity\\brain\\340da9b4-3953-4f3f-9854-5174e0c27de0")
    if os.path.exists(brain_dir):
        import shutil
        shutil.copy(plot_p1, os.path.join(brain_dir, "sopseg_option1_vs_option2_benchmark.png"))
        shutil.copy(plot_master, os.path.join(brain_dir, "sopseg_integrated_master_comparison.png"))
        print("[Artifacts Synced] Plots copied to brain directory.")


# ---------------------------------------------------------------------------
# 7. Main Execution Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Integrated SOPSeg Experiments")
    parser.add_argument("--option", type=str, default="all", choices=["1", "2", "all"], help="Option to run: 1, 2, or all")
    parser.add_argument("--all", action="store_true", help="Run all options (Option 1 and Option 2)")
    parser.add_argument("--source", type=str, default="both", choices=["cut", "anhgoc", "both"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if getattr(args, "all", False):
        args.option = "all"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Integrated Experiments on device: {device}")

    sources = ["cut", "anhgoc"] if args.source == "both" else [args.source]

    opt1_results = {}
    opt2_results = {}

    # 1. Run Option 1 if requested
    if args.option in ["1", "all"]:
        for src in sources:
            model1, eval1 = train_and_eval_option1(source=src, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
            opt1_results[src] = eval1
            export_geotiffs(eval1["pred_scenes"], prefix=f"sopseg_opt1_wave_{src}")

    # 2. Run Option 2 if requested
    if args.option in ["2", "all"]:
        for src in sources:
            model2, eval2 = train_and_eval_option2(source=src, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
            opt2_results[src] = eval2
            export_geotiffs(eval2["pred_scenes"], prefix=f"sopseg_opt2_cd_{src}")

    # 3. Master Benchmark & Visual Plotting
    if args.option == "all" or (len(opt1_results) > 0 and len(opt2_results) > 0):
        # Fill in any missing source if single source was chosen
        cut_opt1 = opt1_results.get("cut", opt1_results.get("anhgoc"))
        anh_opt1 = opt1_results.get("anhgoc", opt1_results.get("cut"))
        cut_opt2 = opt2_results.get("cut", opt2_results.get("anhgoc"))
        anh_opt2 = opt2_results.get("anhgoc", opt2_results.get("cut"))

        compile_benchmark_and_plots(cut_opt1, anh_opt1, cut_opt2, anh_opt2)

    print("\n=======================================================")
    print(" ALL EXPERIMENTS AND BENCHMARKS COMPLETED SUCCESSFULLY! ")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
