import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import discover_scenes, BiTemporalCanGioDataset, SequentialCanGioDataset
from src.train import CombinedSegmentationLoss
from src.evaluate import compute_binary_metrics, stretch_rgb
from src.temporal_models import build_bitemporal_model, build_sequential_model


def plot_bitemporal_evaluation(
    x1_tensor: torch.Tensor,
    x2_tensor: torch.Tensor,
    gt_mask: np.ndarray,
    pred_probs: np.ndarray,
    meta: dict,
    save_path: str,
    threshold: float = 0.5
):
    """
    6-panel visualization for bi-temporal change detection (Option 2):
    1. Pre-image RGB (T1)
    2. Post-image RGB (T2)
    3. Bi-temporal NBR / NDVI delta (|T2 - T1|)
    4. Ground Truth disturbance mask
    5. Predicted disturbance probability
    6. Classification confusion overlay (TP, FP, FN)
    """
    x1 = x1_tensor.detach().cpu().numpy()
    x2 = x2_tensor.detach().cpu().numpy()

    # Bands: Blue=0, Green=1, Red=2, NIR=3
    rgb1 = stretch_rgb(np.stack([x1[2], x1[1], x1[0]], axis=-1))
    rgb2 = stretch_rgb(np.stack([x2[2], x2[1], x2[0]], axis=-1))

    # Delta NBR or Delta NDVI
    # If indices are present, NDVI is at index 6 or computed directly:
    nir1, red1 = x1[3], x1[2]
    nir2, red2 = x2[3], x2[2]
    ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-7)
    ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-7)
    delta_ndvi = ndvi2 - ndvi1

    bin_pred = (pred_probs >= threshold).astype(np.float32)
    gt = (gt_mask >= 0.5).astype(np.float32)

    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    tp = (bin_pred == 1) & (gt == 1)
    fp = (bin_pred == 1) & (gt == 0)
    fn = (bin_pred == 0) & (gt == 1)

    overlay[tp] = [0.0, 0.9, 0.0]  # Green: True Positive
    overlay[fp] = [0.9, 0.1, 0.1]  # Red: False Positive
    overlay[fn] = [0.1, 0.4, 0.9]  # Blue: False Negative

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Option 2: Bi-Temporal Siamese Disturbance Detection\nT1 ({meta.get('t1_date', 'T1')}) -> T2 ({meta.get('t2_date', 'T2')})",
        fontsize=14, fontweight="bold"
    )

    axes[0, 0].imshow(rgb1)
    axes[0, 0].set_title(f"Pre-image RGB ({meta.get('t1_date', 'T1')})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb2)
    axes[0, 1].set_title(f"Post-image RGB ({meta.get('t2_date', 'T2')})")
    axes[0, 1].axis("off")

    im_diff = axes[0, 2].imshow(delta_ndvi, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title(r"Spectral $\Delta$NDVI ($T_2 - T_1$)")
    axes[0, 2].axis("off")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(gt, cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Ground Truth Disturbance (Pos: {int(gt.sum())})")
    axes[1, 0].axis("off")

    im_pred = axes[1, 1].imshow(pred_probs, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Predicted Probability (Thresh: {threshold})")
    axes[1, 1].axis("off")
    fig.colorbar(im_pred, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(rgb2 * 0.4 + overlay * 0.6)
    axes[1, 2].set_title("Overlay (Green: TP, Red: FP, Blue: FN)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_sequential_evaluation(
    x_seq_tensor: torch.Tensor,
    gt_mask: np.ndarray,
    pred_probs: np.ndarray,
    meta: dict,
    save_path: str,
    threshold: float = 0.5
):
    """
    Visualization for Option 3: Sequential ConvLSTM / Temporal Attention.
    """
    seq = x_seq_tensor.detach().cpu().numpy()  # (T, C, H, W)
    T = seq.shape[0]
    dates = meta.get("dates", [f"T{i}" for i in range(T)])

    bin_pred = (pred_probs >= threshold).astype(np.float32)
    gt = (gt_mask >= 0.5).astype(np.float32)

    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    tp = (bin_pred == 1) & (gt == 1)
    fp = (bin_pred == 1) & (gt == 0)
    fn = (bin_pred == 0) & (gt == 1)

    overlay[tp] = [0.0, 0.9, 0.0]
    overlay[fp] = [0.9, 0.1, 0.1]
    overlay[fn] = [0.1, 0.4, 0.9]

    # Create subplots: T images for temporal sequence + GT + Pred + Overlay
    cols = max(3, T)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 4, 8))
    fig.suptitle(
        f"Option 3: Sequential Spatiotemporal Disturbance Modeling\nSequence: {' -> '.join(str(d) for d in dates)}",
        fontsize=14, fontweight="bold"
    )

    # Top row: sequential RGB images
    for t in range(cols):
        ax = axes[0, t]
        if t < T:
            rgb = stretch_rgb(np.stack([seq[t, 2], seq[t, 1], seq[t, 0]], axis=-1))
            ax.imshow(rgb)
            ax.set_title(f"Step {t+1}: {dates[t]}")
        else:
            ax.axis("off")
        ax.axis("off")

    # Bottom row: GT, Prediction, Overlay
    axes[1, 0].imshow(gt, cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Ground Truth (Target: {dates[-1]})")
    axes[1, 0].axis("off")

    im_pred = axes[1, 1].imshow(pred_probs, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Predicted Probability (Thresh: {threshold})")
    axes[1, 1].axis("off")
    fig.colorbar(im_pred, ax=axes[1, 1], fraction=0.046, pad=0.04)

    last_rgb = stretch_rgb(np.stack([seq[-1, 2], seq[-1, 1], seq[-1, 0]], axis=-1))
    axes[1, 2].imshow(last_rgb * 0.4 + overlay * 0.6)
    axes[1, 2].set_title("Overlay (Green: TP, Red: FP, Blue: FN)")
    axes[1, 2].axis("off")

    # Hide extra axes if cols > 3
    for c in range(3, cols):
        axes[1, c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def train_option2_siamese(
    encoder_name: str = "resnet34",
    data_source: str = "cut",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> dict:
    """
    Trains and evaluates Option 2: Siamese 2D U-Net with ResNet-34 or ResNet-50.
    """
    print(f"\n=======================================================")
    print(f"[*] Training Option 2: Siamese U-Net ({encoder_name.upper()}) | Source: {data_source.upper()}")
    print(f"=======================================================")

    all_scenes = discover_scenes(os.path.join(PROJECT_ROOT, "data"), source=data_source)
    # Split: Train on scenes up to 2011, Validate on 2014 scenes
    train_scenes = [s for s in all_scenes if "2014" not in s["year_folder"]]
    val_scenes = [s for s in all_scenes if "2014" in s["year_folder"] or "2011" in s["year_folder"]]

    train_ds = BiTemporalCanGioDataset(train_scenes, pairing_strategy="all", augment=True)
    val_ds = BiTemporalCanGioDataset(val_scenes, pairing_strategy="consecutive", augment=False)

    print(f"Option 2 dataset - Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_bitemporal_model(
        model_type="siamese",
        encoder_name=encoder_name,
        in_channels=9,
        encoder_weights="imagenet"
    ).to(device)

    criterion = CombinedSegmentationLoss(dice_weight=0.5, bce_weight=0.5, pos_weight=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = -1.0
    history = {"train_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for t1, t2, mask, _ in train_loader:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(t1)

        scheduler.step()
        epoch_loss /= max(1, len(train_ds))
        history["train_loss"].append(epoch_loss)

        # Validation
        model.eval()
        val_dice_list, val_iou_list = [], []
        with torch.no_grad():
            for t1, t2, mask, _ in val_loader:
                t1, t2 = t1.to(device), t2.to(device)
                logits = model(t1, t2)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                m_np = mask.numpy()[0, 0]
                metrics = compute_binary_metrics(probs, m_np)
                val_dice_list.append(metrics["dice"])
                val_iou_list.append(metrics["iou"])

        mean_dice = float(np.mean(val_dice_list)) if val_dice_list else 0.0
        mean_iou = float(np.mean(val_iou_list)) if val_iou_list else 0.0
        history["val_dice"].append(mean_dice)
        history["val_iou"].append(mean_iou)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f}")

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            ckpt_path = os.path.join(MODELS_DIR, f"best_option2_siamese_{encoder_name}_{data_source}.pth")
            torch.save(model.state_dict(), ckpt_path)

    # Save visual prediction on first validation sample
    model.eval()
    with torch.no_grad():
        t1, t2, mask, meta = val_ds[0]
        t1_in = t1.unsqueeze(0).to(device)
        t2_in = t2.unsqueeze(0).to(device)
        probs = torch.sigmoid(model(t1_in, t2_in)).cpu().numpy()[0, 0]
        plot_path = os.path.join(PLOTS_DIR, f"eval_option2_siamese_{encoder_name}_{data_source}.png")
        plot_bitemporal_evaluation(t1, t2, mask[0].numpy(), probs, meta, plot_path)
        final_metrics = compute_binary_metrics(probs, mask[0].numpy())

    return {
        "model": f"Siamese UNet ({encoder_name})",
        "option": "Option 2 (Bi-Temporal Siamese)",
        "encoder": encoder_name,
        "data_source": data_source,
        "best_val_dice": round(best_val_dice, 4),
        "final_metrics": final_metrics,
        "plot_path": plot_path,
        "checkpoint": ckpt_path
    }


def train_option3_sequential(
    model_type: str = "convlstm",  # "convlstm" or "attention"
    encoder_name: str = "resnet34",
    data_source: str = "cut",
    seq_len: int = 3,
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 3e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> dict:
    """
    Trains and evaluates Option 3: Sequential Spatiotemporal (ConvLSTM / Temporal Attention) with ResNet-34 or ResNet-50.
    """
    display_type = "ConvLSTM" if model_type == "convlstm" else "Temporal Attention"
    print(f"\n=======================================================")
    print(f"[*] Training Option 3: {display_type} ({encoder_name.upper()}) | Source: {data_source.upper()}")
    print(f"=======================================================")

    all_scenes = discover_scenes(os.path.join(PROJECT_ROOT, "data"), source=data_source)
    # Split: Train up to 2011, Val on 2014
    train_scenes = [s for s in all_scenes if "2014" not in s["year_folder"]]
    val_scenes = [s for s in all_scenes if "2014" in s["year_folder"] or "2011" in s["year_folder"]]

    train_ds = SequentialCanGioDataset(train_scenes, seq_len=seq_len, augment=True)
    val_ds = SequentialCanGioDataset(val_scenes, seq_len=seq_len, augment=False)

    print(f"Option 3 dataset - Train seqs: {len(train_ds)}, Val seqs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_sequential_model(
        model_type=model_type,
        encoder_name=encoder_name,
        in_channels=9,
        encoder_weights="imagenet"
    ).to(device)

    criterion = CombinedSegmentationLoss(dice_weight=0.5, bce_weight=0.5, pos_weight=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = -1.0
    history = {"train_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for seq, mask, _ in train_loader:
            seq, mask = seq.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(seq)

        scheduler.step()
        epoch_loss /= max(1, len(train_ds))
        history["train_loss"].append(epoch_loss)

        # Validation
        model.eval()
        val_dice_list, val_iou_list = [], []
        with torch.no_grad():
            for seq, mask, _ in val_loader:
                seq = seq.to(device)
                logits = model(seq)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                m_np = mask.numpy()[0, 0]
                metrics = compute_binary_metrics(probs, m_np)
                val_dice_list.append(metrics["dice"])
                val_iou_list.append(metrics["iou"])

        mean_dice = float(np.mean(val_dice_list)) if val_dice_list else 0.0
        mean_iou = float(np.mean(val_iou_list)) if val_iou_list else 0.0
        history["val_dice"].append(mean_dice)
        history["val_iou"].append(mean_iou)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f}")

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            ckpt_path = os.path.join(MODELS_DIR, f"best_option3_{model_type}_{encoder_name}_{data_source}.pth")
            torch.save(model.state_dict(), ckpt_path)

    # Save visual prediction on first validation sample
    model.eval()
    with torch.no_grad():
        seq, mask, meta = val_ds[0]
        seq_in = seq.unsqueeze(0).to(device)
        probs = torch.sigmoid(model(seq_in)).cpu().numpy()[0, 0]
        plot_path = os.path.join(PLOTS_DIR, f"eval_option3_{model_type}_{encoder_name}_{data_source}.png")
        plot_sequential_evaluation(seq, mask[0].numpy(), probs, meta, plot_path)
        final_metrics = compute_binary_metrics(probs, mask[0].numpy())

    return {
        "model": f"{display_type} ({encoder_name})",
        "option": f"Option 3 ({display_type})",
        "encoder": encoder_name,
        "data_source": data_source,
        "best_val_dice": round(best_val_dice, 4),
        "final_metrics": final_metrics,
        "plot_path": plot_path,
        "checkpoint": ckpt_path
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Alternative 2 & 3 for Can Gio Disturbance Segmentation")
    parser.add_argument("--option", choices=["2", "3", "all"], default="all", help="Which option to train: 2 (Siamese), 3 (Sequential), all")
    parser.add_argument("--encoder", choices=["resnet34", "resnet50", "all"], default="all", help="ResNet backbone to evaluate")
    parser.add_argument("--source", choices=["cut", "anhgoc"], default="cut", help="Data source: cut or anhgoc")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    encoders = ["resnet34", "resnet50"] if args.encoder == "all" else [args.encoder]
    all_results = []

    # 1. Run Option 2: Siamese U-Net
    if args.option in ("2", "all"):
        for enc in encoders:
            res = train_option2_siamese(
                encoder_name=enc,
                data_source=args.source,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device
            )
            all_results.append(res)

    # 2. Run Option 3: Sequential ConvLSTM & Temporal Attention
    if args.option in ("3", "all"):
        for enc in encoders:
            # 3A: ConvLSTM
            res_lstm = train_option3_sequential(
                model_type="convlstm",
                encoder_name=enc,
                data_source=args.source,
                seq_len=3,
                epochs=args.epochs,
                batch_size=max(2, args.batch_size // 2),
                device=device
            )
            all_results.append(res_lstm)

            # 3B: Temporal Attention
            res_attn = train_option3_sequential(
                model_type="attention",
                encoder_name=enc,
                data_source=args.source,
                seq_len=3,
                epochs=args.epochs,
                batch_size=max(2, args.batch_size // 2),
                device=device
            )
            all_results.append(res_attn)

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, f"temporal_alternatives_summary_{args.source}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("[*] SUMMARY COMPARISON OF ALTERNATIVES (OPTION 2 & OPTION 3)")
    print("=" * 70)
    print(f"{'Model Architecture':<36} | {'Encoder':<10} | {'Best Val Dice':<14} | {'Val IoU':<10}")
    print("-" * 70)
    for r in all_results:
        iou = r['final_metrics']['iou']
        print(f"{r['model']:<36} | {r['encoder']:<10} | {r['best_val_dice']:<14.4f} | {iou:<10.4f}")
    print("=" * 70)
    print(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
