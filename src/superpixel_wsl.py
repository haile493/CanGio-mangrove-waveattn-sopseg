import os
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation as seg
from skimage.color import rgb2gray
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.models import get_model_from_config
from src.evaluate import evaluate_model, compute_binary_metrics


def generate_superpixels(
    image: np.ndarray,
    n_segments: int = 100,
    compactness: float = 10.0,
    method: str = "slic"
) -> np.ndarray:
    """
    Computes superpixel partition on an image (C, H, W) or (H, W, 3).
    Methods: 'slic' or 'felzenszwalb'
    Returns: 2D integer array of shape (H, W) with segment IDs.
    """
    if image.ndim == 3 and image.shape[0] in (3, 6, 9):
        # Extract false-color composite NIR-Red-Green for strong botanical boundary contrast
        # Channels: 0: Blue, 1: Green, 2: Red, 3: NIR
        nir = image[3] if image.shape[0] > 3 else image[2]
        red = image[2]
        green = image[1]
        rgb_like = np.stack([nir, red, green], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 3:
        rgb_like = image
    else:
        rgb_like = np.repeat(image[0, :, :, None], 3, axis=-1)

    # Normalize to [0, 1]
    rgb_norm = (rgb_like - rgb_like.min()) / (rgb_like.max() - rgb_like.min() + 1e-7)

    if method == "slic":
        segments = seg.slic(rgb_norm, n_segments=n_segments, compactness=compactness, start_label=0)
    elif method == "felzenszwalb":
        segments = seg.felzenszwalb(rgb_norm, scale=100, sigma=0.5, min_size=50)
    else:
        raise ValueError(f"Unknown superpixel method: {method}")

    return segments


def sample_query_points(
    gt_mask: np.ndarray,
    n_points: int = 50,
    balance_classes: bool = True
) -> List[Tuple[int, int, int]]:
    """
    Simulates manual sparse query point annotations (Rivier et al. Section 3.1).
    Returns list of (y, x, class_id) points.
    class_id: 0 = Background, 1 = Disturbance ("vùng gãy đổ")
    """
    mask_2d = gt_mask.squeeze()
    pos_coords = np.argwhere(mask_2d >= 0.5)
    neg_coords = np.argwhere(mask_2d < 0.5)

    points = []
    if balance_classes and len(pos_coords) > 0:
        # Distribute points equally between foreground and background
        n_pos = min(n_points // 2, len(pos_coords))
        n_neg = n_points - n_pos

        if n_pos > 0:
            idx_pos = np.random.choice(len(pos_coords), size=n_pos, replace=False)
            for idx in idx_pos:
                y, x = pos_coords[idx]
                points.append((int(y), int(x), 1))

        if n_neg > 0:
            idx_neg = np.random.choice(len(neg_coords), size=n_neg, replace=False)
            for idx in idx_neg:
                y, x = neg_coords[idx]
                points.append((int(y), int(x), 0))
    else:
        # Uniform random point sampling
        total_pixels = mask_2d.size
        h, w = mask_2d.shape
        flat_indices = np.random.choice(total_pixels, size=min(n_points, total_pixels), replace=False)
        for f_idx in flat_indices:
            y = f_idx // w
            x = f_idx % w
            cls_id = int(mask_2d[y, x] >= 0.5)
            points.append((int(y), int(x), cls_id))

    return points


def extend_points_to_superpixels(
    superpixels: np.ndarray,
    query_points: List[Tuple[int, int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagates query point annotations across containing superpixel regions (Rivier et al. Section 3.3).
    Returns:
    - pseudo_label (H, W): Propagated class label for labeled superpixels (0 or 1).
    - active_mask (H, W): Binary indicator (1 = labeled superpixel, 0 = unlabeled / ignored).
    """
    h, w = superpixels.shape
    pseudo_label = np.zeros((h, w), dtype=np.float32)
    active_mask = np.zeros((h, w), dtype=np.float32)

    # Group points by superpixel segment ID
    seg_points: Dict[int, List[int]] = {}
    for y, x, cls_id in query_points:
        seg_id = int(superpixels[y, x])
        if seg_id not in seg_points:
            seg_points[seg_id] = []
        seg_points[seg_id].append(cls_id)

    # Majority voting propagation
    for seg_id, class_list in seg_points.items():
        # Count classes
        pos_count = sum(1 for c in class_list if c == 1)
        neg_count = sum(1 for c in class_list if c == 0)

        chosen_class = 1.0 if pos_count >= neg_count else 0.0
        seg_mask = (superpixels == seg_id)

        pseudo_label[seg_mask] = chosen_class
        active_mask[seg_mask] = 1.0  # Mark superpixel as labeled

    return pseudo_label, active_mask


class WeightedMaskedLoss(nn.Module):
    """
    Weighted Masked Loss as proposed in Rivier et al. Section 3.4:
    L = (1/N) * sum_i (1/C) * sum_c [ m_{i,c} * (w_{l_{i,c}} - p_{l_{i,c}})^2 ]
    Adapted for binary segmentation with logits and class-rebalancing weights.
    """
    def __init__(self, pos_weight: float = 5.0, use_dice: bool = True):
        super().__init__()
        self.pos_weight = pos_weight
        self.use_dice = use_dice
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        active_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: (B, 1, H, W)
        pseudo_labels: (B, 1, H, W)
        active_masks: (B, 1, H, W)
        """
        probs = torch.sigmoid(logits)

        # 1. Weighted Masked MSE as in the paper
        diff_sq = (pseudo_labels - probs) ** 2
        
        # Apply class weights: higher penalty on rare positive disturbance class
        class_weights = torch.where(pseudo_labels >= 0.5, self.pos_weight, 1.0)
        masked_mse = active_masks * class_weights * diff_sq
        
        num_active = torch.clamp(active_masks.sum(), min=1.0)
        mse_loss = masked_mse.sum() / num_active

        if self.use_dice:
            # Masked Soft Dice on active regions only
            intersection = (probs * pseudo_labels * active_masks).sum(dim=(2, 3))
            denominator = (probs * active_masks + pseudo_labels * active_masks).sum(dim=(2, 3)) + 1e-7
            dice_loss = 1.0 - (2.0 * intersection / denominator).mean()
            return 0.5 * mse_loss + 0.5 * dice_loss
        else:
            return mse_loss


class SuperpixelWSLDataset(Dataset):
    """
    Weakly-Supervised Dataset that generates superpixel pseudo-labels on the fly
    from simulated sparse query points.
    """
    def __init__(
        self,
        base_dataset: CanGioDataset,
        n_points: int = 50,
        n_segments: int = 100,
        balance_classes: bool = True
    ):
        self.base_dataset = base_dataset
        self.n_points = n_points
        self.n_segments = n_segments
        self.balance_classes = balance_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img_tensor, gt_mask_tensor, meta = self.base_dataset[idx]
        img_np = img_tensor.numpy()
        gt_np = gt_mask_tensor.numpy()

        # 1. Generate superpixels
        superpixels = generate_superpixels(img_np, n_segments=self.n_segments)

        # 2. Sample sparse query points
        points = sample_query_points(gt_np, n_points=self.n_points, balance_classes=self.balance_classes)

        # 3. Propagate to superpixel pseudo-labels
        pseudo_label, active_mask = extend_points_to_superpixels(superpixels, points)

        pseudo_tensor = torch.from_numpy(pseudo_label).unsqueeze(0).float()
        active_tensor = torch.from_numpy(active_mask).unsqueeze(0).float()

        return img_tensor, pseudo_tensor, active_tensor, gt_mask_tensor, meta


def train_superpixel_wsl(
    config: Optional[Config] = None,
    n_points: int = 50,
    n_segments: int = 100,
    num_epochs: int = 15
) -> Tuple[nn.Module, Dict]:
    """
    Trains U-Net using Superpixel-Assisted Weakly Supervised Learning (Study 1).
    """
    if config is None:
        config = Config()
    config.num_epochs = num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Study 1: Superpixel WSL] Training with {n_points} query points per scene, {n_segments} superpixels")

    all_scenes = discover_scenes(config.data_dir, source=config.data_source)
    gt_scenes = [s for s in all_scenes if s.get("groundtruth_path") and os.path.exists(s["groundtruth_path"])]

    train_scenes = [s for s in gt_scenes if not any(vy in s["year_folder"] for vy in config.val_years)]
    val_scenes = [s for s in gt_scenes if any(vy in s["year_folder"] for vy in config.val_years)]

    base_train = CanGioDataset(train_scenes, spectral_bands=config.spectral_bands, include_indices=config.include_indices, indices=config.indices, augment=True, study_bounds=config.study_bounds)
    wsl_train_dataset = SuperpixelWSLDataset(base_train, n_points=n_points, n_segments=n_segments)
    val_dataset = CanGioDataset(val_scenes, spectral_bands=config.spectral_bands, include_indices=config.include_indices, indices=config.indices, augment=False, study_bounds=config.study_bounds)

    train_loader = DataLoader(wsl_train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = get_model_from_config(config).to(device)
    criterion = WeightedMaskedLoss(pos_weight=10.0, use_dice=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    best_val_dice = -1.0
    best_model_path = os.path.join(config.models_dir, "best_superpixel_wsl.pth")

    history = {"train_loss": [], "val_iou": [], "val_dice": [], "val_precision": [], "val_recall": []}

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_losses = []
        for _ in range(4):  # 4 augment passes per epoch
            for images, pseudo_masks, active_masks, _, _ in train_loader:
                images = images.to(device)
                pseudo_masks = pseudo_masks.to(device)
                active_masks = active_masks.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, pseudo_masks, active_masks)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(avg_loss)

        # Evaluate on ground truth validation scenes
        val_metrics, _ = evaluate_model(model, val_loader, device=device, threshold=config.threshold)
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        print(
            f"[WSL Epoch {epoch:02d}/{config.num_epochs:02d}] "
            f"Loss: {avg_loss:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Dice/F1: {val_metrics['dice']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_model_path)

    print(f"[Study 1: Superpixel WSL] Training finished. Best Validation Dice: {best_val_dice:.4f}")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    return model, history
