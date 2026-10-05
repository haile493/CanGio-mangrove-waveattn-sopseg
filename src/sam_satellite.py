import os
import urllib.request
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor
import segmentation_models_pytorch as smp

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.evaluate import compute_binary_metrics


SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"


def ensure_sam_checkpoint(models_dir: str = MODELS_DIR) -> str:
    """Ensures SAM ViT-B weights exist locally, downloading if necessary."""
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, "sam_vit_b_01ec64.pth")
    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 300 * 1024 * 1024:
        print(f"Downloading SAM ViT-B checkpoint to {ckpt_path}...")
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, ckpt_path)
        print("Download complete.")
    return ckpt_path


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer applied to linear projections (Hu et al. / Scribano et al.).
    Computes: y = W_0 x + (B * A) x * (alpha / r)
    """
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze base pretrained weight
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_f = linear.in_features
        out_f = linear.out_features

        # LoRA trainable decomposition matrices
        device = linear.weight.device
        dtype = linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(in_f, rank, device=device, dtype=dtype) * (1.0 / np.sqrt(in_f)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def apply_lora_to_sam(sam_model: nn.Module, rank: int = 8, alpha: float = 16.0):
    """
    Applies LoRA to SAM's ViT image encoder attention layers and configures
    trainable parameters (Scribano et al. Section 4.1).
    - Image Encoder: Base weights frozen, LoRA layers trainable.
    - Prompt Encoder: Completely frozen.
    - Mask Decoder: Trainable (full fine-tuning).
    """
    # 1. Freeze entire SAM first
    for param in sam_model.parameters():
        param.requires_grad = False

    # 2. Inject LoRA into ViT image encoder self-attention qkv projection layers
    lora_layers = []
    for block in sam_model.image_encoder.blocks:
        if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)
            lora_layers.append(block.attn.qkv)

    # 3. Unfreeze mask decoder as per Paper 2 strategy (ii)
    for param in sam_model.mask_decoder.parameters():
        param.requires_grad = True

    # Count parameters
    total_params = sum(p.numel() for p in sam_model.parameters())
    trainable_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    print(f"[SAM LoRA Setup] Total params: {total_params:,} | Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")


def prepare_satellite_for_sam(
    image_tensor: torch.Tensor,
    target_size: int = 1024
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Converts multi-spectral satellite tensor (C, H, W) to SAM's expected format:
    Extracts high-contrast False-Color composite (NIR, Red, Green) and resizes to 1024x1024.
    """
    c, h, w = image_tensor.shape
    nir = image_tensor[3] if c > 3 else image_tensor[2]
    red = image_tensor[2]
    green = image_tensor[1]

    # False-color 3-channel composite
    composite = torch.stack([nir, red, green], dim=0)  # (3, H, W)
    
    # Scale to ImageNet standard [0, 255] then normalize
    c_min = composite.amin(dim=(1, 2), keepdim=True)
    c_max = composite.amax(dim=(1, 2), keepdim=True)
    comp_norm = (composite - c_min) / (c_max - c_min + 1e-7) * 255.0

    # Resize to SAM's native 1024x1024 input
    comp_resized = F.interpolate(comp_norm.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False)
    
    # Standard SAM pixel normalization (pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375])
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=image_tensor.device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=image_tensor.device).view(1, 3, 1, 1)
    sam_input = (comp_resized - pixel_mean) / pixel_std

    return sam_input.squeeze(0), (h, w)


def generate_prompts(
    gt_mask: np.ndarray,
    orig_size: Tuple[int, int],
    target_size: int = 1024,
    prompt_type: str = "single_point"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates point prompts scaled to SAM's 1024x1024 coordinate system.
    Returns:
    - points: (N, 2) in (X, Y) format.
    - labels: (N,) with 1 for foreground, 0 for background.
    """
    h_orig, w_orig = orig_size
    scale_y = target_size / h_orig
    scale_x = target_size / w_orig

    mask_2d = gt_mask.squeeze()
    pos_coords = np.argwhere(mask_2d >= 0.5)
    neg_coords = np.argwhere(mask_2d < 0.5)

    points = []
    labels = []

    if len(pos_coords) == 0:
        # Fallback center point
        points.append([w_orig // 2 * scale_x, h_orig // 2 * scale_y])
        labels.append(0)
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

    if prompt_type == "single_point":
        # Centroid of positive disturbance region
        mean_y, mean_x = pos_coords.mean(axis=0)
        points.append([mean_x * scale_x, mean_y * scale_y])
        labels.append(1)

    elif prompt_type == "multi_point":
        # 4 positive points + 2 negative border points (Scribano et al. Section 4.2)
        n_pos = min(4, len(pos_coords))
        n_neg = min(2, len(neg_coords))

        idx_pos = np.random.choice(len(pos_coords), size=n_pos, replace=False)
        for idx in idx_pos:
            y, x = pos_coords[idx]
            points.append([x * scale_x, y * scale_y])
            labels.append(1)

        idx_neg = np.random.choice(len(neg_coords), size=n_neg, replace=False)
        for idx in idx_neg:
            y, x = neg_coords[idx]
            points.append([x * scale_x, y * scale_y])
            labels.append(0)

    elif prompt_type == "grid":
        # Regular automatic grid (e.g. 8x8)
        grid_steps = 8
        xs = np.linspace(20, target_size - 20, grid_steps)
        ys = np.linspace(20, target_size - 20, grid_steps)
        for gy in ys:
            for gx in xs:
                orig_y = int(gy / scale_y)
                orig_x = int(gx / scale_x)
                orig_y = min(max(orig_y, 0), h_orig - 1)
                orig_x = min(max(orig_x, 0), w_orig - 1)
                lbl = int(mask_2d[orig_y, orig_x] >= 0.5)
                points.append([gx, gy])
                labels.append(lbl)

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def train_sam_lora(
    config: Optional[Config] = None,
    num_epochs: int = 8,
    lora_rank: int = 8,
    lr: float = 1e-4,
    prompt_type: str = "multi_point"
) -> Tuple[nn.Module, Dict]:
    """
    Fine-tunes SAM on Can Gio satellite scenes using LoRA and point prompts (Study 2).
    """
    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Study 2: SAM + LoRA] Setting up SAM ViT-B on {device}...")

    ckpt_path = ensure_sam_checkpoint(config.models_dir)
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=ckpt_path).to(device)

    # Apply LoRA to image encoder and unfreeze mask decoder
    apply_lora_to_sam(sam, rank=lora_rank, alpha=16.0)

    all_scenes = discover_scenes(config.data_dir, source=config.data_source)
    gt_scenes = [s for s in all_scenes if s.get("groundtruth_path") and os.path.exists(s["groundtruth_path"])]
    train_scenes = [s for s in gt_scenes if not any(vy in s["year_folder"] for vy in config.val_years)]
    val_scenes = [s for s in gt_scenes if any(vy in s["year_folder"] for vy in config.val_years)]

    train_dataset = CanGioDataset(train_scenes, spectral_bands=config.spectral_bands, include_indices=False, augment=True, study_bounds=config.study_bounds)
    val_dataset = CanGioDataset(val_scenes, spectral_bands=config.spectral_bands, include_indices=False, augment=False, study_bounds=config.study_bounds)

    optimizer = torch.optim.AdamW([p for p in sam.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    best_ckpt_path = os.path.join(config.models_dir, "best_sam_lora.pth")
    history = {"train_loss": [], "val_iou": [], "val_dice": []}

    print(f"[Study 2: SAM + LoRA] Starting training for {num_epochs} epochs with prompt type: '{prompt_type}'...")

    for epoch in range(1, num_epochs + 1):
        sam.train()
        epoch_losses = []

        for idx in range(len(train_dataset)):
            img_tensor, mask_tensor, meta = train_dataset[idx]
            gt_np = mask_tensor.numpy().squeeze()
            orig_size = (img_tensor.shape[1], img_tensor.shape[2])

            # Prepare image and points
            sam_img, _ = prepare_satellite_for_sam(img_tensor)
            sam_img = sam_img.unsqueeze(0).to(device)

            pts_np, lbls_np = generate_prompts(gt_np, orig_size, target_size=1024, prompt_type=prompt_type)
            pts_tensor = torch.from_numpy(pts_np).unsqueeze(0).to(device)
            lbls_tensor = torch.from_numpy(lbls_np).unsqueeze(0).to(device)

            optimizer.zero_grad()

            # SAM forward pass
            image_embeddings = sam.image_encoder(sam_img)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=(pts_tensor, lbls_tensor),
                boxes=None,
                masks=None
            )
            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            # Resize predicted mask back to original 124x124 resolution
            pred_mask = F.interpolate(low_res_masks, size=orig_size, mode="bilinear", align_corners=False)
            gt_target = mask_tensor.unsqueeze(0).to(device)

            loss = 0.5 * dice_loss_fn(pred_mask, gt_target) + 0.5 * bce_loss_fn(pred_mask, gt_target)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(avg_loss)

        # Validation step
        val_metrics = evaluate_sam(sam, val_dataset, device, prompt_type=prompt_type)
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])

        print(
            f"[SAM LoRA Epoch {epoch:02d}/{num_epochs:02d}] "
            f"Loss: {avg_loss:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Dice/F1: {val_metrics['dice']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            # Save only trainable LoRA + decoder weights to keep file compact
            trainable_state = {k: v for k, v in sam.state_dict().items() if "lora_" in k or "mask_decoder" in k}
            torch.save(trainable_state, best_ckpt_path)

    print(f"[Study 2: SAM + LoRA] Fine-tuning finished. Best Validation Dice: {best_val_dice:.4f}")
    return sam, history


def evaluate_sam(
    sam_model: nn.Module,
    val_dataset: CanGioDataset,
    device: torch.device,
    prompt_type: str = "multi_point",
    threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluates SAM model on validation scenes."""
    sam_model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            img_tensor, mask_tensor, _ = val_dataset[idx]
            gt_np = mask_tensor.numpy().squeeze()
            orig_size = (img_tensor.shape[1], img_tensor.shape[2])

            sam_img, _ = prepare_satellite_for_sam(img_tensor)
            sam_img = sam_img.unsqueeze(0).to(device)

            pts_np, lbls_np = generate_prompts(gt_np, orig_size, target_size=1024, prompt_type=prompt_type)
            pts_tensor = torch.from_numpy(pts_np).unsqueeze(0).to(device)
            lbls_tensor = torch.from_numpy(lbls_np).unsqueeze(0).to(device)

            image_embeddings = sam_model.image_encoder(sam_img)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=(pts_tensor, lbls_tensor),
                boxes=None,
                masks=None
            )
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            pred_mask = F.interpolate(low_res_masks, size=orig_size, mode="bilinear", align_corners=False)
            prob = torch.sigmoid(pred_mask).cpu().numpy().squeeze()

            all_preds.append(prob.flatten())
            all_targets.append(gt_np.flatten())

    cat_preds = np.concatenate(all_preds)
    cat_targets = np.concatenate(all_targets)
    return compute_binary_metrics(cat_preds, cat_targets, threshold=threshold)
