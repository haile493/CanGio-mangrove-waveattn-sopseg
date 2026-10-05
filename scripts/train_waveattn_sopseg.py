"""
Training Script for WaveAttn SOPSeg
"High-Precision Mangrove Canopy Disturbance Mapping in the Can Gio Biosphere Reserve
via Frequency-Aware Prompted Foundation Models"

Usage:
    python scripts/train_waveattn_sopseg.py --source full --epochs 12 --batch_size 4 --lr 1e-4
    python scripts/train_waveattn_sopseg.py --source cut --epochs 12 --batch_size 4 --lr 1e-4
    python scripts/train_waveattn_sopseg.py --source both --epochs 12
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DATA_DIR, MODELS_DIR
from src.dataset import CanGioDataset, discover_scenes
from src.sopseg import (
    RegionAdaptiveMagnification,
    OrientedPromptMechanism,
    apply_position_embedding_interpolation,
    SOPSegMultiTaskLoss
)
from src.sopseg_enhanced import (
    WaveAttnSOPSegModel,
    EnhancedSOPSegModel
)
from scripts.run_sopseg_integrated_cd import (
    EnhancedSOPSegDataset,
    extract_enhanced_instances,
    evaluate_option1
)
from segment_anything import sam_model_registry


def build_waveattn_model(
    sam_checkpoint: str,
    device: str = "cuda",
    lora_rank: int = 8,
    lora_alpha: float = 16.0
) -> WaveAttnSOPSegModel:
    """Builds and initializes the WaveAttn SOPSeg model with pre-trained SAM ViT backbone."""
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found at: {sam_checkpoint}")

    base_sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    apply_position_embedding_interpolation(base_sam, target_size=256)
    
    model = WaveAttnSOPSegModel(
        base_sam=base_sam,
        shallow_layer_idx=3,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        guidance_channels=4
    )
    return model.to(device)


def train_single_source(
    source: str = "full",
    epochs: int = 12,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = MODELS_DIR
) -> str:
    """Trains WaveAttn SOPSeg on either 'full' (full radiometric scenes) or 'cut' (cut chips)."""
    src_folder = "anhgoc" if source.lower() in ("full", "full_scenes", "anhgoc") else "cut"
    display_title = "Full Radiometric Scenes" if src_folder == "anhgoc" else "Cut Chips (124x124)"
    
    print("\n" + "=" * 70)
    print(f" TRAINING WAVEATTN SOPSEG ON: {display_title.upper()}")
    print("=" * 70)

    sam_ckpt = os.path.join(save_dir, "sam_vit_b_01ec64.pth")
    model = build_waveattn_model(sam_ckpt, device=device)
    loss_fn = SOPSegMultiTaskLoss(lambda_iou=5.0)
    ram = RegionAdaptiveMagnification(s_in=256)

    # Discover and split dataset
    scenes = discover_scenes(DATA_DIR, source=src_folder)
    val_years = ["2011", "2014"]
    train_scenes = [s for s in scenes if not any(y in str(s.get("date", "")) for y in val_years)]
    val_scenes = [s for s in scenes if any(y in str(s.get("date", "")) for y in val_years)]

    dataset_cls = CanGioDataset(scenes=scenes, require_gt=True)
    
    print(f"[*] Extracting training instance patches for {src_folder}...")
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

    print(f"[Dataset] Train Instances: {len(train_instances)} | Validation Instances: {len(val_instances)}")

    train_ds = EnhancedSOPSegDataset(train_instances, ram, is_training=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_iou = 0.0
    save_filename = f"best_sopseg_enhanced_{src_folder}.pth"
    save_path = os.path.join(save_dir, save_filename)

    for ep in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {ep:02d}/{epochs:02d}", leave=False)
        for batch in pbar:
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_epoch_loss / max(1, len(train_loader))

        # Validation at epoch end
        val_eval = evaluate_option1(model, val_scene_records, ram, device=device)
        curr_iou = val_eval["global_metrics"]["iou"]
        curr_biou = val_eval["global_metrics"]["biou"]
        curr_dice = val_eval["global_metrics"]["dice"]

        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val IoU: {curr_iou:.4f} | Dice: {curr_dice:.4f} | BIoU: {curr_biou:.4f}")

        if curr_iou > best_iou:
            best_iou = curr_iou
            torch.save(model.state_dict(), save_path)
            print(f"  --> [Saved Best Model] Val IoU improved to {best_iou:.4f} -> {save_path}")

    print(f"\n[Completed] Training on {display_title} finished. Best Val IoU: {best_iou:.4f}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train WaveAttn SOPSeg on mangrove disturbance datasets.")
    parser.add_argument("--source", type=str, default="full", choices=["full", "cut", "both"], help="Data source regime: full scenes or cut chips")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs (default: 12)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for instance patches (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--save_dir", type=str, default=MODELS_DIR, help="Directory to save trained model weights")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    sources = ["full", "cut"] if args.source == "both" else [args.source]
    for s in sources:
        train_single_source(
            source=s,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            save_dir=args.save_dir
        )


if __name__ == "__main__":
    main()
