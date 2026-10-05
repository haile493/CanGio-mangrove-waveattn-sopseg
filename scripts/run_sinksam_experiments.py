"""
SinkSAM-Net Execution and Comprehensive Benchmark Suite
======================================================
Implements Rafaeli et al. (ISPRS Journal of Photogrammetry and Remote Sensing, 2025):
"SinkSAM-Net: Knowledge-driven self-supervised sinkhole segmentation using topographic priors
and Segment Anything Model" (https://github.com/osherr1996/SinkSAM)
Applied to Can Gio Disturbance Segmentation on both 'Cut' and 'Anhgoc' datasets.

Workflow:
  1. Topographic & Closed Depression candidate delineation
  2. Coordinate-Wise Bounding Box Jittering (CWBJ)
  3. SAM Monte Carlo ensemble segmentation simulations (N=20)
  4. Epistemic uncertainty computation & Polsby-Popper shape filtering
  5. Training lightweight target model (EffV2-UNet) with Weighted Soft Dice Loss + BCE
  6. Self-training refinement loop
  7. Prompt-free evaluation on held-out test scenes (2011 & 2014)
  8. Master benchmark comparison against previous methods
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import json
import argparse
import shutil
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.measure import regionprops
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
from src.dataset import CanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, compute_boundary_iou, stretch_rgb
from src.sinksam import (
    compute_closed_depressions,
    run_sinksam_monte_carlo,
    filter_pseudo_labels_shape,
    SinkSAMLoss,
    build_effv2_unet
)
from segment_anything import sam_model_registry, SamPredictor

ARTIFACT_DIR = r"C:\Users\hai49\.gemini\antigravity\brain\bb83dad2-1132-4783-9c2c-48cca30de8b0"


# ---------------------------------------------------------------------------
# 1. Pseudo-Label Generation via SinkSAM-Net Stages 1-5
# ---------------------------------------------------------------------------

def generate_sinksam_pseudo_labels(
    scenes_list: List[Dict],
    source: str,
    device: str = "cuda",
    num_samples: int = 20,
    jitter_pixels: int = 3
) -> List[Dict[str, Any]]:
    """
    Executes Stages 1-5 of SinkSAM-Net for all training scenes.
    Generates high-confidence pseudo-labels via CWBJ, SAM Monte Carlo, and Polsby-Popper filtering.
    """
    print(f"\n=======================================================")
    print(f" [SinkSAM-Net] Generating Pseudo-Labels for source: {source.upper()}")
    print(f"=======================================================")
    
    # Load SAM ViT-B predictor
    ckpt_path = os.path.join(MODELS_DIR, "sam_vit_b_01ec64.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {ckpt_path}")
        
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    ds = CanGioDataset(scenes_list, include_indices=True, augment=False)
    pseudo_records = []

    for idx in range(len(ds)):
        img_tensor, gt_tensor, meta = ds[idx]
        scene_id = meta["scene_id"]
        gt_np = (gt_tensor.numpy().squeeze() > 0.5).astype(np.uint8)

        # Stage 1 & 2: Delineate closed depressions
        depr_map, cand_mask, boxes = compute_closed_depressions(
            img_tensor,
            closing_size=9,
            depth_thresh=0.12,
            min_area=3,
            max_area=400
        )

        # Stage 3 & 4: SAM Monte Carlo simulations with CWBJ
        mean_mask, unc_map = run_sinksam_monte_carlo(
            predictor=predictor,
            image_tensor=img_tensor,
            boxes=boxes,
            num_samples=num_samples,
            jitter_pixels=jitter_pixels,
            device=device
        )

        # Stage 5: Polsby-Popper circularity and compactness shape filtering
        pseudo_label = filter_pseudo_labels_shape(
            mean_mask=mean_mask,
            depression_cand_mask=cand_mask,
            threshold=0.5,
            min_area=4,
            pp_min=0.12
        )

        # Compute pseudo-label quality against ground truth (for diagnostic reporting)
        m_pseudo = compute_binary_metrics(pseudo_label.flatten(), gt_np.flatten())
        print(f"  Scene [{meta['year_folder']}] {scene_id}: {len(boxes)} boxes -> Pseudo-label: IoU={m_pseudo['iou']:.4f}, Dice={m_pseudo['dice']:.4f}, Prec={m_pseudo['precision']:.4f}, Rec={m_pseudo['recall']:.4f}")

        pseudo_records.append({
            "img_tensor": img_tensor,
            "pseudo_label": torch.from_numpy(pseudo_label).unsqueeze(0).float(),
            "gt_np": gt_np,
            "meta": meta,
            "depr_map": depr_map,
            "cand_mask": cand_mask,
            "mean_mask": mean_mask,
            "unc_map": unc_map,
            "boxes": boxes
        })

    # Release SAM memory from GPU
    del predictor
    del sam
    torch.cuda.empty_cache()
    return pseudo_records


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset for Target Model Training
# ---------------------------------------------------------------------------

class SinkSAMPseudoDataset(Dataset):
    """Dataset fed into EffV2-UNet target model during self-supervised training."""
    def __init__(self, records: List[Dict[str, Any]], augment: bool = True):
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img = rec["img_tensor"].clone()
        mask = rec["pseudo_label"].clone()

        if self.augment:
            # Random horizontal / vertical flips
            if np.random.rand() > 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if np.random.rand() > 0.5:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[1])
            # Random 90 deg rotation
            k = int(np.random.choice([0, 1, 2, 3]))
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[1, 2])

        return img, mask


# ---------------------------------------------------------------------------
# 3. Target Model Training (EffV2-UNet)
# ---------------------------------------------------------------------------

def train_effv2_target_model(
    train_records: List[Dict[str, Any]],
    source: str,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 8e-4,
    device: str = "cuda"
) -> nn.Module:
    """
    Trains the lightweight EffV2-UNet target model on SinkSAM pseudo-labels.
    """
    print(f"\n[SinkSAM-Net] Training EffV2-UNet on {source.upper()} pseudo-labels ({epochs} epochs)...")
    model = build_effv2_unet(in_channels=9, encoder_name="timm-efficientnet-b0").to(device)
    model.train()

    dataset = SinkSAMPseudoDataset(train_records, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = SinkSAMLoss(w_s=1.0, w_n=0.07, bce_weight=1.0, dice_weight=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch [{ep:02d}/{epochs:02d}] - SinkSAM Loss: {avg_loss:.4f} (lr: {scheduler.get_last_lr()[0]:.6f})")

    # Save checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"sinksam_effv2_{source}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  [Saved Checkpoint] {save_path}")
    return model


# ---------------------------------------------------------------------------
# 4. Self-Training Refinement Loop (Stage 6)
# ---------------------------------------------------------------------------

def self_training_refinement(
    model: nn.Module,
    train_records: List[Dict[str, Any]],
    source: str,
    device: str = "cuda",
    refine_epochs: int = 15
) -> nn.Module:
    """
    Executes the self-training iteration:
      1. EffV2-UNet predicts on training scenes
      2. High-confidence predictions generate updated bounding boxes
      3. SAM with CWBJ re-refines masks
      4. EffV2-UNet is retrained on updated pseudo-labels
    """
    print(f"\n[SinkSAM-Net] Executing Self-Training Iteration for {source.upper()}...")
    model.eval()

    ckpt_path = os.path.join(MODELS_DIR, "sam_vit_b_01ec64.pth")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    refined_records = []
    with torch.no_grad():
        for rec in train_records:
            img = rec["img_tensor"].unsqueeze(0).to(device)
            pred_logits = model(img)
            pred_prob = torch.sigmoid(pred_logits).cpu().numpy().squeeze()
            bin_pred = (pred_prob >= 0.45).astype(np.uint8)

            # Extract updated bounding boxes from predicted clusters
            labeled, _ = label(bin_pred)
            new_boxes = []
            for r in regionprops(labeled):
                if 4 <= r.area <= 400:
                    y1, x1, y2, x2 = r.bbox
                    new_boxes.append([max(0, x1 - 2), max(0, y1 - 2), min(124, x2 + 2), min(124, y2 + 2)])

            # If model found boxes, re-prompt SAM with CWBJ; else keep previous
            if new_boxes:
                mean_m, _ = run_sinksam_monte_carlo(
                    predictor=predictor,
                    image_tensor=rec["img_tensor"],
                    boxes=new_boxes,
                    num_samples=15,
                    jitter_pixels=3,
                    device=device
                )
                refined_pseudo = filter_pseudo_labels_shape(
                    mean_mask=mean_m,
                    depression_cand_mask=rec["cand_mask"],
                    threshold=0.5,
                    min_area=4,
                    pp_min=0.12
                )
            else:
                refined_pseudo = rec["pseudo_label"].squeeze().numpy()

            updated_rec = dict(rec)
            updated_rec["pseudo_label"] = torch.from_numpy(refined_pseudo).unsqueeze(0).float()
            refined_records.append(updated_rec)

    del predictor
    del sam
    torch.cuda.empty_cache()

    # Retrain EffV2-UNet on refined pseudo-labels
    retrained_model = train_effv2_target_model(
        train_records=refined_records,
        source=f"{source}_selftrained",
        epochs=refine_epochs,
        batch_size=4,
        lr=5e-4,
        device=device
    )
    return retrained_model


# ---------------------------------------------------------------------------
# 5. Prompt-Free Evaluation on Test Scenes
# ---------------------------------------------------------------------------

def evaluate_sinksam_target_model(
    model: nn.Module,
    test_scenes: List[Dict],
    source: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Evaluates the prompt-free EffV2-UNet target model on the held-out test scenes (2011 & 2014).
    Computes IoU, Dice/F1, Precision, Recall, Accuracy, Boundary-IoU (BIoU), and confusion counts.
    Exports georeferenced prediction GeoTIFFs.
    """
    print(f"\n=======================================================")
    print(f" [SinkSAM-Net Evaluation] Testing on {len(test_scenes)} scenes ({source.upper()})")
    print(f"=======================================================")
    model.eval()

    ds = CanGioDataset(test_scenes, include_indices=True, augment=False)
    all_preds = []
    all_gts = []
    per_scene_results = []
    biou_list = []

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(ds)):
            img_tensor, gt_tensor, meta = ds[idx]
            gt_np = (gt_tensor.numpy().squeeze() > 0.5).astype(np.uint8)

            img_input = img_tensor.unsqueeze(0).to(device)
            logits = model(img_input)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            bin_pred = (probs >= 0.5).astype(np.uint8)

            all_preds.append(bin_pred.flatten())
            all_gts.append(gt_np.flatten())

            sc_metrics = compute_binary_metrics(bin_pred.flatten(), gt_np.flatten(), threshold=0.5)
            sc_biou = compute_boundary_iou(bin_pred, gt_np, threshold=0.5)
            sc_metrics["biou"] = float(sc_biou)
            biou_list.append(sc_biou)

            sc_metrics["scene_id"] = meta["scene_id"]
            sc_metrics["year_folder"] = meta["year_folder"]
            sc_metrics["date"] = meta["date"]
            sc_metrics["sensor"] = meta["sensor"]
            per_scene_results.append(sc_metrics)

            print(f"  [{meta['year_folder']}] {meta['scene_id']}: IoU={sc_metrics['iou']:.4f}, BIoU={sc_biou:.4f}, Dice={sc_metrics['dice']:.4f}, Prec={sc_metrics['precision']:.4f}, Rec={sc_metrics['recall']:.4f}")

            # Export georeferenced GeoTIFF
            out_tif = os.path.join(PREDICTIONS_DIR, f"sinksam_effv2_{source}_{meta['scene_id']}_pred.tif")
            crs = meta.get("crs", "EPSG:32648")
            trans = meta.get("transform")
            h, w = bin_pred.shape

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
                dst.write(bin_pred.astype(np.float32), 1)
                dst.write(probs.astype(np.float32), 2)
                dst.set_band_description(1, "SinkSAM_BinaryMask")
                dst.set_band_description(2, "SinkSAM_DisturbanceProb")

    # Global pooled metrics across all test pixels
    pooled_preds = np.concatenate(all_preds)
    pooled_gts = np.concatenate(all_gts)
    global_metrics = compute_binary_metrics(pooled_preds, pooled_gts, threshold=0.5)
    global_metrics["biou"] = float(np.mean(biou_list))

    print(f"\n--- [Global Pooled Test Results for {source.upper()}] ---")
    print(f"  IoU:       {global_metrics['iou']:.4f}")
    print(f"  BIoU:      {global_metrics['biou']:.4f}")
    print(f"  Dice (F1): {global_metrics['dice']:.4f}")
    print(f"  Precision: {global_metrics['precision']:.4f}")
    print(f"  Recall:    {global_metrics['recall']:.4f}")
    print(f"  Accuracy:  {global_metrics['accuracy']:.4f}")
    print(f"  TP: {global_metrics['tp']} | FP: {global_metrics['fp']} | FN: {global_metrics['fn']} | TN: {global_metrics['tn']}")

    return {
        "global_metrics": global_metrics,
        "per_scene_results": per_scene_results
    }


# ---------------------------------------------------------------------------
# 6. Benchmark Comparison and Visualization Generation
# ---------------------------------------------------------------------------

def generate_benchmark_visualizations(
    cut_results: Dict[str, Any],
    anhgoc_results: Dict[str, Any],
    sample_records_cut: List[Dict[str, Any]],
    sample_records_anhgoc: List[Dict[str, Any]],
    output_dir: str = PLOTS_DIR
):
    """
    Generates:
      1. Comparative bar charts: SinkSAM Cut vs. Anhgoc
      2. Master benchmark comparison against all prior paradigms
      3. Multi-panel qualitative visual analysis demonstrating SinkSAM stages
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # Chart 1: SinkSAM-Net Cut vs. Anhgoc
    # -------------------------------------------------------------
    metrics = ["iou", "biou", "dice", "precision", "recall"]
    metric_labels = ["IoU", "Boundary IoU (BIoU)", "Dice (F1)", "Precision", "Recall"]
    
    c_vals = [cut_results["global_metrics"][m] for m in metrics]
    a_vals = [anhgoc_results["global_metrics"][m] for m in metrics]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
    x = np.arange(2)
    width = 0.5

    for idx, (m_name, title) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        vals = [c_vals[idx], a_vals[idx]]
        bars = ax.bar(["Cut Chips", "Anhgoc (Raw)"], vals, color=["#1f77b4", "#ff7f0e"], width=width, edgecolor="black", linewidth=1.0)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("SinkSAM-Net Performance: Pre-Cut Chips vs. Original Full Scenes (Anhgoc)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    p1 = os.path.join(output_dir, "sinksam_cut_vs_anhgoc.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(ARTIFACT_DIR, "sinksam_cut_vs_anhgoc.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {p1}")

    # -------------------------------------------------------------
    # Chart 2: Master Benchmark across All Paradigms (Anhgoc)
    # -------------------------------------------------------------
    paradigms = [
        "Superpixel WSL",
        "Fully-Sup U-Net",
        "SAM + LoRA",
        "SAM2-CD",
        "SOPSeg Enhanced",
        "SOPSeg-CD",
        "SinkSAM-Net (Ours)"
    ]
    
    # Established Anhgoc benchmarks from project history
    # WSL: 0.0147, UNet: 0.3782, SAM+LoRA: 0.5413, SAM2-CD: 0.5620, SOPSeg Enh: 0.7576, SOPSeg-CD: 0.5207
    iou_master = [0.0147, 0.3782, 0.5413, 0.5620, 0.7576, 0.5207, anhgoc_results["global_metrics"]["iou"]]
    biou_master = [0.0090, 0.2740, 0.4180, 0.4950, 0.9634, 0.6956, anhgoc_results["global_metrics"]["biou"]]
    dice_master = [0.0290, 0.5488, 0.7024, 0.7196, 0.8621, 0.6848, anhgoc_results["global_metrics"]["dice"]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.0))
    colors = ["#7f7f7f", "#8c564b", "#9467bd", "#17becf", "#1f77b4", "#2ca02c", "#d62728"]

    for idx, (title, data) in enumerate([("Intersection over Union (IoU)", iou_master),
                                         ("Boundary IoU (BIoU)", biou_master),
                                         ("Dice / F1-Score", dice_master)]):
        ax = axes[idx]
        bars = ax.bar(paradigms, data, color=colors, edgecolor="black", width=0.6)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(0, 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_xticklabels(paradigms, rotation=25, ha="right", fontsize=9, fontweight="bold")

        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle("Master Benchmark: Self-Supervised SinkSAM-Net vs. All Existing Segmentation Paradigms (Anhgoc)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    p2 = os.path.join(output_dir, "sinksam_master_benchmark.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(ARTIFACT_DIR, "sinksam_master_benchmark.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {p2}")

    # -------------------------------------------------------------
    # Chart 3: Qualitative 6-Panel Demonstration of SinkSAM Stages
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    rows = [("Cut Chips", sample_records_cut[0]), ("Anhgoc (Raw)", sample_records_anhgoc[0])]

    for row_idx, (r_name, rec) in enumerate(rows):
        img_np = rec["img_tensor"].numpy()
        nir = img_np[3]
        red = img_np[2]
        green = img_np[1]
        false_color = np.stack([nir, red, green], axis=-1)
        false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min() + 1e-7)

        # 1. False Color
        axes[row_idx, 0].imshow(false_color)
        axes[row_idx, 0].set_title(f"{r_name}\nNIR-R-G False Color", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 2. Closed Depression Depth Map
        axes[row_idx, 1].imshow(rec["depr_map"], cmap="magma")
        axes[row_idx, 1].set_title("Stage 1-2\nClosed Depressions", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # 3. CWBJ Bounding Box Prompts
        img_boxes = false_color.copy()
        for b in rec["boxes"][:15]:
            cv2.rectangle(img_boxes, (b[0], b[1]), (b[2], b[3]), (0, 1, 0), 1)
        axes[row_idx, 2].imshow(img_boxes)
        axes[row_idx, 2].set_title(f"Stage 3\nCWBJ Prompt Boxes ({len(rec['boxes'])})", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # 4. SAM Monte Carlo Mean Mask
        axes[row_idx, 3].imshow(rec["mean_mask"], cmap="viridis")
        axes[row_idx, 3].set_title("Stage 4\nSAM Mean Mask (N=20)", fontsize=10, fontweight="bold")
        axes[row_idx, 3].axis("off")

        # 5. Polsby-Popper Filtered Pseudo-Label
        pseudo = rec["pseudo_label"].squeeze().numpy()
        axes[row_idx, 4].imshow(pseudo, cmap="gray")
        axes[row_idx, 4].set_title("Stage 5\nPolsby-Popper Refined", fontsize=10, fontweight="bold")
        axes[row_idx, 4].axis("off")

        # 6. Ground Truth
        axes[row_idx, 5].imshow(rec["gt_np"], cmap="gray")
        axes[row_idx, 5].set_title("Ground Truth\nVùng Gãy Đổ", fontsize=10, fontweight="bold")
        axes[row_idx, 5].axis("off")

    fig.suptitle("SinkSAM-Net Pipeline Execution: Step-by-Step Visualization across Stages 1–5", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    p3 = os.path.join(output_dir, "sinksam_qualitative_stages.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(ARTIFACT_DIR, "sinksam_qualitative_stages.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {p3}")


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SinkSAM-Net Experiment Suite for Can Gio")
    parser.add_argument("--epochs", type=int, default=18, help="Target model training epochs")
    parser.add_argument("--samples", type=int, default=20, help="SAM Monte Carlo simulations N")
    parser.add_argument("--jitter", type=int, default=3, help="CWBJ jitter range in pixels")
    parser.add_argument("--no_self_training", action="store_true", help="Skip iterative self-training")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f" SinkSAM-Net: Self-Supervised Segmentation Suite")
    print(f" Target Model: EffV2-UNet (timm-efficientnet-b0)")
    print(f" Device: {device} | Epochs: {args.epochs} | MC Samples: {args.samples}")
    print(f"=======================================================")

    results_master = {}

    # Discover scenes
    scenes_cut = discover_scenes(DATA_DIR, source="cut")
    scenes_anhgoc = discover_scenes(DATA_DIR, source="anhgoc")

    train_cut = [s for s in scenes_cut if not any(y in s["year_folder"] for y in ["2011", "2014"]) and s.get("groundtruth_path")]
    test_cut = [s for s in scenes_cut if any(y in s["year_folder"] for y in ["2011", "2014"]) and s.get("groundtruth_path")]

    train_anhgoc = [s for s in scenes_anhgoc if not any(y in s["year_folder"] for y in ["2011", "2014"]) and s.get("groundtruth_path")]
    test_anhgoc = [s for s in scenes_anhgoc if any(y in s["year_folder"] for y in ["2011", "2014"]) and s.get("groundtruth_path")]

    print(f"Dataset split:")
    print(f"  Train: {len(train_cut)} scenes (2006-2010)")
    print(f"  Test:  {len(test_cut)} scenes (2011 & 2014)")

    # -------------------------------------------------------------
    # 1. Run Pipeline on 'Cut' (Cropped Chips)
    # -------------------------------------------------------------
    print("\n>>> STAGE I: SinkSAM-Net on CUT CHIPS <<<")
    cut_pseudo_records = generate_sinksam_pseudo_labels(
        scenes_list=train_cut,
        source="cut",
        device=device,
        num_samples=args.samples,
        jitter_pixels=args.jitter
    )

    model_cut = train_effv2_target_model(
        train_records=cut_pseudo_records,
        source="cut",
        epochs=args.epochs,
        device=device
    )

    if not args.no_self_training:
        model_cut = self_training_refinement(
            model=model_cut,
            train_records=cut_pseudo_records,
            source="cut",
            device=device,
            refine_epochs=12
        )

    cut_eval = evaluate_sinksam_target_model(
        model=model_cut,
        test_scenes=test_cut,
        source="cut",
        device=device
    )
    results_master["cut"] = cut_eval

    # -------------------------------------------------------------
    # 2. Run Pipeline on 'Anhgoc' (Original Full Scenes)
    # -------------------------------------------------------------
    print("\n>>> STAGE II: SinkSAM-Net on ANHGOC (RAW FULL SCENES) <<<")
    anhgoc_pseudo_records = generate_sinksam_pseudo_labels(
        scenes_list=train_anhgoc,
        source="anhgoc",
        device=device,
        num_samples=args.samples,
        jitter_pixels=args.jitter
    )

    model_anhgoc = train_effv2_target_model(
        train_records=anhgoc_pseudo_records,
        source="anhgoc",
        epochs=args.epochs,
        device=device
    )

    if not args.no_self_training:
        model_anhgoc = self_training_refinement(
            model=model_anhgoc,
            train_records=anhgoc_pseudo_records,
            source="anhgoc",
            device=device,
            refine_epochs=12
        )

    anhgoc_eval = evaluate_sinksam_target_model(
        model=model_anhgoc,
        test_scenes=test_anhgoc,
        source="anhgoc",
        device=device
    )
    results_master["anhgoc"] = anhgoc_eval

    # -------------------------------------------------------------
    # 3. Save Benchmark Results JSON
    # -------------------------------------------------------------
    out_json = os.path.join(RESULTS_DIR, "sinksam_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results_master, f, indent=2)
    print(f"\n[Saved JSON] Master SinkSAM results saved to: {out_json}")

    # Copy to artifact directory
    shutil.copy(out_json, os.path.join(ARTIFACT_DIR, "sinksam_benchmark_results.json"))

    # -------------------------------------------------------------
    # 4. Generate Visualizations & Master Benchmark Plots
    # -------------------------------------------------------------
    generate_benchmark_visualizations(
        cut_results=cut_eval,
        anhgoc_results=anhgoc_eval,
        sample_records_cut=cut_pseudo_records,
        sample_records_anhgoc=anhgoc_pseudo_records,
        output_dir=PLOTS_DIR
    )

    print("\n=======================================================")
    print(" SinkSAM-Net Pipeline Execution & Benchmarking Complete!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
