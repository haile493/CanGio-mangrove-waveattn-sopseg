import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SAM2_CD_REPO_DIR = os.path.join(PROJECT_ROOT, "external_repos", "SAM2-CD", "SAM2-CD-main")
if SAM2_CD_REPO_DIR not in sys.path:
    sys.path.insert(0, SAM2_CD_REPO_DIR)

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import BiTemporalCanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, stretch_rgb
from models.build_sam import build_sam2


class AutoPadWrapper(nn.Module):
    """
    Pads spatial dimensions (H, W) to a multiple of `multiple` (e.g. 32 or 64)
    and unpads model outputs back to (H, W).
    """
    def __init__(self, model: nn.Module, multiple: int = 32):
        super().__init__()
        self.model = model
        self.multiple = multiple

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        pad_h = (self.multiple - (h % self.multiple)) % self.multiple
        pad_w = (self.multiple - (w % self.multiple)) % self.multiple

        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_pad = x

        outputs = self.model(x_pad)
        if isinstance(outputs, (tuple, list)):
            return tuple(out[..., :h, :w] for out in outputs)
        return outputs[..., :h, :w]


class SAM2CDWrapper(nn.Module):
    """
    SAM2-CD architecture adapted for Can Gio bi-temporal imagery.
    Takes separate (T1, T2) tensors and processes them through the
    SAM2-CD pyramid encoder, ASG, MSFFM, WF, and GLCAM decoder.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        in_channels: int = 3,
        device: str = "cuda"
    ):
        super().__init__()
        if config_path is None:
            config_path = os.path.join(SAM2_CD_REPO_DIR, "sam2_configs", "sam2_cd_hiera_s.yaml")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(MODELS_DIR, "sam2_hiera_small.pt")

        self.device = device
        self.in_channels = in_channels

        # Base SAM2-CD model
        base_sam2_cd = build_sam2(config_path, checkpoint_path, device=device)
        self.base_model = AutoPadWrapper(base_sam2_cd, multiple=32)

        # Optional channel adapter if input has > 3 channels (e.g. 9 channels: 6 bands + 3 indices)
        if in_channels != 3:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        else:
            self.channel_adapter = nn.Identity()

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for bi-temporal input pair:
        t1: (B, C, H, W)
        t2: (B, C, H, W)
        Returns:
            mask_p2, mask_p3, mask_p4: Logit maps at different decoder levels.
        """
        t1_adapted = self.channel_adapter(t1)
        t2_adapted = self.channel_adapter(t2)

        # Concatenate along batch dimension as expected by SAM2-CD: [B*2, 3, H, W]
        combined_batch = torch.cat([t1_adapted, t2_adapted], dim=0)
        mask_p2, mask_p3, mask_p4 = self.base_model(combined_batch)
        return mask_p2, mask_p3, mask_p4


def train_sam2_cd(
    source: str = "cut",
    in_channels: int = 3,
    num_epochs: int = 15,
    batch_size: int = 2,
    lr: float = 5e-5,
    device: str = "cuda",
    save_ckpt: bool = True
) -> Tuple[SAM2CDWrapper, Dict[str, float]]:
    """
    Fine-tunes SAM2-CD on the Can Gio dataset for either 'cut' or 'anhgoc'.
    """
    print(f"\n=======================================================")
    print(f" TRAINING SAM2-CD ({source.upper()} IMAGERY, {in_channels} CHANNELS)")
    print(f"=======================================================")

    scenes = discover_scenes(os.path.join(PROJECT_ROOT, "data"), source=source)
    val_years = ["2011", "2014"]
    train_scenes = [s for s in scenes if not any(y in s["year_folder"] for y in val_years)]
    val_scenes = [s for s in scenes if any(y in s["year_folder"] for y in val_years) and s.get("groundtruth_path")]

    # If in_channels == 3, use RGB bands only (B, G, R)
    spectral_bands = ["blue", "green", "red"] if in_channels == 3 else ["blue", "green", "red", "nir", "swir1", "swir2"]
    include_indices = (in_channels != 3)

    train_ds = BiTemporalCanGioDataset(
        train_scenes,
        pairing_strategy="all",
        spectral_bands=spectral_bands,
        include_indices=include_indices,
        augment=True
    )
    val_ds = BiTemporalCanGioDataset(
        val_scenes,
        pairing_strategy="all",
        spectral_bands=spectral_bands,
        include_indices=include_indices,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f"Training pairs: {len(train_ds)} | Validation pairs: {len(val_ds)}")

    model = SAM2CDWrapper(in_channels=in_channels, device=device).to(device)

    # Train decoder, MSFFM, WF, and adapter while keeping Hiera trunk largely frozen or low lr
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_iou = 0.0
    best_metrics = {}
    save_path = os.path.join(MODELS_DIR, f"best_sam2_cd_{source}.pth")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for t1, t2, gt_mask, _ in train_loader:
            t1, t2, gt_mask = t1.to(device).float(), t2.to(device).float(), gt_mask.to(device).float()
            optimizer.zero_grad()

            p2, p3, p4 = model(t1, t2)

            # Combined BCE + Dice loss across pyramid predictions
            bce_loss = F.binary_cross_entropy_with_logits(p2, gt_mask) + \
                       0.5 * F.binary_cross_entropy_with_logits(p3, gt_mask) + \
                       0.25 * F.binary_cross_entropy_with_logits(p4, gt_mask)

            prob2 = torch.sigmoid(p2)
            intersection = (prob2 * gt_mask).sum()
            dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (prob2.sum() + gt_mask.sum() + 1e-6)

            loss = bce_loss + dice_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_loader))

        # Evaluate on validation pairs
        val_metrics = evaluate_sam2_cd(model, val_loader, device=device)
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] Loss: {avg_train_loss:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['dice']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_metrics = val_metrics
            if save_ckpt:
                torch.save(model.state_dict(), save_path)
                print(f"  --> Saved new best checkpoint to {save_path} (IoU: {best_iou:.4f})")

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return model, best_metrics


def evaluate_sam2_cd(
    model: SAM2CDWrapper,
    val_loader: DataLoader,
    device: str = "cuda",
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluates SAM2-CD model over the provided DataLoader.
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for t1, t2, gt_mask, _ in val_loader:
            t1, t2 = t1.to(device).float(), t2.to(device).float()
            p2, _, _ = model(t1, t2)
            probs = torch.sigmoid(p2).squeeze(1).cpu().numpy()
            targets = gt_mask.squeeze(1).numpy()

            all_preds.append(probs)
            all_targets.append(targets)

    preds_cat = np.concatenate(all_preds, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)
    metrics = compute_binary_metrics(preds_cat, targets_cat, threshold=threshold)
    return metrics
