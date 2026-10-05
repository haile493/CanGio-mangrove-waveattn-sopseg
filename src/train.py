import os
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.models import get_model_from_config
from src.evaluate import evaluate_model


class VietnameseUrbanCompositeLoss(nn.Module):
    """
    Composite Loss Function for Semantic Segmentation from Dinh et al. (2026), Eq. (3) & (10):
    L_sem = 0.5 * L_Dice + 0.3 * L_Focal (gamma=2.0) + 0.2 * L_Boundary
    Designed specifically to overcome Vietnamese dense urban morphology challenges:
    - Dice Loss: handles severe background-building class imbalance.
    - Focal Loss: forces learning hard-to-distinguish spectral similarities.
    - Boundary Loss: directly penalizes deviations along boundary transition corridors.
    """
    def __init__(self, w_dice: float = 0.5, w_focal: float = 0.3, w_boundary: float = 0.2, gamma: float = 2.0, alpha: float = 0.75, eps: float = 1e-6):
        super().__init__()
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_boundary = w_boundary
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def extract_boundary(self, x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        pad = kernel_size // 2
        d = torch.nn.functional.max_pool2d(x, kernel_size, stride=1, padding=pad)
        e = -torch.nn.functional.max_pool2d(-x, kernel_size, stride=1, padding=pad)
        return (d - e).clamp(0.0, 1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # 1. Dice Loss (per-batch foreground overlap)
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)
        intersection = (p_flat * t_flat).sum()
        l_dice = 1.0 - (2.0 * intersection + self.eps) / (p_flat.sum() + t_flat.sum() + self.eps)

        # 2. Focal Loss: - alpha_t * (1 - pt)^gamma * log(pt)
        p_clamped = probs.clamp(self.eps, 1.0 - self.eps)
        pt = torch.where(targets == 1.0, p_clamped, 1.0 - p_clamped)
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        focal_weight = (1.0 - pt) ** self.gamma
        l_focal = - (alpha_t * focal_weight * torch.log(pt)).mean()

        # 3. Boundary Loss (1 - F1_boundary)
        p_b = self.extract_boundary(probs)
        t_b = self.extract_boundary(targets)
        p_b_flat = p_b.view(-1)
        t_b_flat = t_b.view(-1)
        b_inter = (p_b_flat * t_b_flat).sum()
        f1_boundary = (2.0 * b_inter + self.eps) / (p_b_flat.sum() + t_b_flat.sum() + self.eps)
        l_boundary = 1.0 - f1_boundary

        return self.w_dice * l_dice + self.w_focal * l_focal + self.w_boundary * l_boundary



class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for highly imbalanced binary remote sensing segmentation:
    Loss = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss(pos_weight)
    """
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, pos_weight: float = 10.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_device_pos = self.bce_loss.pos_weight.to(logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=bce_device_pos)
        dice = self.dice_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce



def train_model(config: Optional[Config] = None, num_epochs: Optional[int] = None) -> Tuple[nn.Module, Dict]:
    if config is None:
        config = Config()
    if num_epochs is not None:
        config.num_epochs = num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Discover all scenes
    all_scenes = discover_scenes(config.data_dir, source=config.data_source)
    print(f"Discovered {len(all_scenes)} total scenes in {config.data_dir} (source={config.data_source})")

    # 2. Filter scenes with ground truth
    gt_scenes = [s for s in all_scenes if s.get("groundtruth_path") and os.path.exists(s["groundtruth_path"])]
    print(f"Found {len(gt_scenes)} scenes with ground truth masks.")

    # 3. Train / Validation split
    # By default, use years 2011 and 2014 for validation, rest for training
    train_scenes = [s for s in gt_scenes if not any(vy in s["year_folder"] for vy in config.val_years)]
    val_scenes = [s for s in gt_scenes if any(vy in s["year_folder"] for vy in config.val_years)]

    # Fallback if split yields empty set
    if not train_scenes:
        train_scenes = gt_scenes[:len(gt_scenes)//2]
        val_scenes = gt_scenes[len(gt_scenes)//2:]

    print(f"Train scenes ({len(train_scenes)}): {[s['scene_id'] for s in train_scenes]}")
    print(f"Val scenes ({len(val_scenes)}): {[s['scene_id'] for s in val_scenes]}")

    # Build datasets and loaders
    train_dataset = CanGioDataset(
        scenes=train_scenes,
        spectral_bands=config.spectral_bands,
        include_indices=config.include_indices,
        indices=config.indices,
        augment=True,
        require_gt=True,
        study_bounds=config.study_bounds
    )
    val_dataset = CanGioDataset(
        scenes=val_scenes,
        spectral_bands=config.spectral_bands,
        include_indices=config.include_indices,
        indices=config.indices,
        augment=False,
        require_gt=True,
        study_bounds=config.study_bounds
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 4. Instantiate model
    model = get_model_from_config(config).to(device)

    # 5. Loss, Optimizer, Scheduler
    criterion = CombinedSegmentationLoss(dice_weight=config.dice_weight, bce_weight=config.focal_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    best_val_dice = -1.0
    best_model_path = os.path.join(config.models_dir, f"best_{config.arch}_{config.encoder_name}.pth")
    final_model_path = os.path.join(config.models_dir, f"final_{config.arch}_{config.encoder_name}.pth")

    history = {
        "train_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_precision": [],
        "val_recall": []
    }

    print("\nStarting Training Pipeline...")
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_losses = []
        
        # We run multiple augment passes per epoch to increase sample variety
        steps_per_epoch = 4
        for _ in range(steps_per_epoch):
            for images, masks, _ in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(avg_train_loss)

        # Validation step
        val_metrics, _ = evaluate_model(model, val_loader, device=device, threshold=config.threshold)
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        curr_dice = val_metrics["dice"]
        print(
            f"Epoch [{epoch:02d}/{config.num_epochs:02d}] "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Dice/F1: {curr_dice:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f}"
        )

        if curr_dice > best_val_dice:
            best_val_dice = curr_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "config": config.__dict__
            }, best_model_path)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Validation Dice: {best_val_dice:.4f}")

    # Save final model
    torch.save(model.state_dict(), final_model_path)

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.num_epochs + 1), history["train_loss"], 'r-o', label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.num_epochs + 1), history["val_dice"], 'b-s', label="Val Dice (F1)")
    plt.plot(range(1, config.num_epochs + 1), history["val_iou"], 'g-^', label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.grid(True)
    plt.legend()

    curve_path = os.path.join(config.plots_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()

    # Save history json
    history_path = os.path.join(config.results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Load best weights before returning
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with Dice: {checkpoint['val_dice']:.4f}")

    return model, history


if __name__ == "__main__":
    cfg = Config(num_epochs=15)
    train_model(cfg)
