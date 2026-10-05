import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.models import build_study3_model, STUDY3_MODEL_CONFIGS
from src.train import VietnameseUrbanCompositeLoss
from src.evaluate import compute_binary_metrics, stretch_rgb
from src.spatial_error_analysis import (
    analyze_spatial_errors,
    aggregate_spatial_errors,
    plot_spatial_error_shift
)


def get_dataloaders(source: str, batch_size: int = 4, val_years: List[str] = ["2011", "2014"]):
    """Prepares training and validation DataLoaders for Cut or Anhgoc imagery."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    all_scenes = discover_scenes(data_dir, source=source)
    gt_scenes = [s for s in all_scenes if s.get("groundtruth_path") and os.path.exists(s["groundtruth_path"])]

    train_scenes = [s for s in gt_scenes if not any(vy in s["year_folder"] for vy in val_years)]
    val_scenes = [s for s in gt_scenes if any(vy in s["year_folder"] for vy in val_years)]

    if not train_scenes:
        train_scenes = gt_scenes[:len(gt_scenes)//2]
        val_scenes = gt_scenes[len(gt_scenes)//2:]

    train_ds = CanGioDataset(train_scenes, augment=True, require_gt=True)
    val_ds = CanGioDataset(val_scenes, augment=False, require_gt=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_scenes, val_scenes


def evaluate_study3_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict]]:
    """Evaluates model computing standard metrics and spatial error distributions."""
    model.eval()
    all_preds = []
    all_targets = []
    spatial_error_records = []
    visual_records = []

    with torch.no_grad():
        for batch_idx, (images, masks, metas) in enumerate(val_loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)
            targets_np = masks.numpy()  # (B, 1, H, W)

            b_size = images.shape[0]
            for i in range(b_size):
                p_scene = probs[i, 0]
                t_scene = targets_np[i, 0]
                bin_pred = (p_scene >= threshold).astype(np.float32)
                bin_gt = (t_scene >= 0.5).astype(np.float32)

                all_preds.append(p_scene.flatten())
                all_targets.append(t_scene.flatten())

                # Spatial error analysis
                s_err = analyze_spatial_errors(bin_pred, bin_gt)
                spatial_error_records.append(s_err)

                # Store sample for plotting
                meta_item = {k: v[i] if isinstance(v, list) else v for k, v in metas.items()}
                if len(visual_records) < 4:
                    visual_records.append({
                        "image": images[i].cpu(),
                        "gt": bin_gt,
                        "pred": p_scene,
                        "meta": meta_item,
                        "spatial_error": s_err
                    })

    cat_preds = np.concatenate(all_preds)
    cat_targets = np.concatenate(all_targets)
    global_metrics = compute_binary_metrics(cat_preds, cat_targets, threshold=threshold)
    agg_spatial_errors = aggregate_spatial_errors(spatial_error_records)

    return global_metrics, agg_spatial_errors, visual_records


def finetune_study3_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4
) -> Tuple[nn.Module, Dict]:
    """
    Localized parameter fine-tuning using VietnameseUrbanCompositeLoss:
    L_sem = 0.5 * Dice + 0.3 * Focal + 0.2 * Boundary
    """
    criterion = VietnameseUrbanCompositeLoss(w_dice=0.5, w_focal=0.3, w_boundary=0.2, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_dice = -1.0
    best_weights = None
    history = {"train_loss": [], "val_dice": [], "val_iou": []}

    steps_per_epoch = 4
    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            for images, masks, _ in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(losses))
        history["train_loss"].append(avg_loss)

        # Validation
        val_metrics, _, _ = evaluate_study3_model(model, val_loader, device=device)
        curr_dice = val_metrics["dice"]
        curr_iou = val_metrics["iou"]
        history["val_dice"].append(curr_dice)
        history["val_iou"].append(curr_iou)

        if curr_dice > best_val_dice:
            best_val_dice = curr_dice
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"      Epoch [{epoch:02d}/{num_epochs:02d}] Loss: {avg_loss:.4f} | Val IoU: {curr_iou:.4f} | Val Dice (F1): {curr_dice:.4f}")

    if best_weights:
        model.load_state_dict(best_weights)
        print(f"      Restored best checkpoint (Dice: {best_val_dice:.4f})")

    return model, history


def plot_prediction_spatial_errors(
    visual_sample: Dict,
    save_path: str,
    title: str = "Spatial Error Analysis"
):
    """Visualizes satellite input, GT mask, prediction, and 4-tier spatial error breakdown."""
    img_np = visual_sample["image"].numpy()
    gt = visual_sample["gt"]
    pred = visual_sample["pred"]
    bin_pred = (pred >= 0.5).astype(np.float32)

    red = img_np[2]
    green = img_np[1]
    blue = img_np[0]
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_disp = stretch_rgb(rgb)

    masks = visual_sample["spatial_error"].get("masks", {})
    h, w = gt.shape
    error_vis = np.zeros((h, w, 3), dtype=np.float32)

    # Green for True Positives
    tp = (bin_pred == 1) & (gt == 1)
    error_vis[tp] = [0.1, 0.8, 0.2]

    # Orange for Under-segmentation
    if "under_segmentation" in masks:
        error_vis[masks["under_segmentation"]] = [0.95, 0.5, 0.0]

    # Magenta for Boundary Error
    if "boundary" in masks:
        error_vis[masks["boundary"]] = [0.9, 0.1, 0.6]

    # Purple for False Negative
    if "false_negative" in masks:
        error_vis[masks["false_negative"]] = [0.4, 0.2, 0.8]

    # Cyan/Lime for False Positive
    if "false_positive" in masks:
        error_vis[masks["false_positive"]] = [0.2, 0.8, 0.9]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(rgb_disp)
    axes[0].set_title("True Color (RGB)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Disturbance", fontweight="bold")
    axes[1].axis("off")

    im_pred = axes[2].imshow(pred, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Prediction Probability", fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im_pred, ax=axes[2], fraction=0.046, pad=0.04)

    blended = rgb_disp * 0.35 + error_vis * 0.65
    axes[3].imshow(blended)
    axes[3].set_title("Spatial Errors (Dinh et al.)\nOrange:Under | Mag:Bound | Purp:FN | Cyan:FP", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_cut_vs_anhgoc_study3(results: Dict, save_path: str):
    """
    Renders grouped bar charts comparing all 4 models across Cut vs. Anhgoc imagery.
    """
    model_keys = [k for k in STUDY3_MODEL_CONFIGS if ("cut" in results and k in results["cut"]) or ("anhgoc" in results and k in results["anhgoc"])]
    if not model_keys:
        return
    model_labels = [STUDY3_MODEL_CONFIGS[k]["display_name"] for k in model_keys]
    metrics = ["IoU", "Dice (F1)", "Precision", "Recall"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(model_labels))
    width = 0.35

    for idx, m_key in enumerate(metrics):
        metric_field = "iou" if m_key == "IoU" else ("dice" if m_key == "Dice (F1)" else m_key.lower())
        vals_cut = [results.get("cut", {}).get(mk, {}).get("finetuned", {}).get("metrics", {}).get(metric_field, 0.0) for mk in model_keys]
        vals_anhgoc = [results.get("anhgoc", {}).get(mk, {}).get("finetuned", {}).get("metrics", {}).get(metric_field, 0.0) for mk in model_keys]

        ax = axes[idx]
        r1 = ax.bar(x - width/2, vals_cut, width, label="Cut Chips", color="#1f77b4", edgecolor="black", alpha=0.85)
        r2 = ax.bar(x + width/2, vals_anhgoc, width, label="Anhgoc (Full)", color="#ff7f0e", edgecolor="black", alpha=0.85)

        ax.set_title(m_key, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)


        for rect in r1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
        for rect in r2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color="#b34700", fontweight="bold")

        if idx == 0:
            ax.legend(frameon=True, loc="upper right")

    fig.suptitle("Study 3: Localized Fine-Tuning Performance (Cut Chips vs. Anhgoc Scenes)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Study 3 Cut vs Anhgoc benchmark chart saved to: {save_path}")


def run_study3_pipeline(
    sources: List[str] = ["cut", "anhgoc"],
    models_to_run: Optional[List[str]] = None,
    num_epochs: int = 10,
    lr: float = 1e-4
) -> Dict:
    """
    Executes the full Study 3 experimental framework from Dinh et al. (2026):
    - Phase 1: Zero-shot / Pretrained evaluation & Domain Gap (Delta_gap) quantification
    - Phase 2: Localized fine-tuning with Composite Loss (Delta_FT)
    - Phase 3: Error-Centric Spatial Analysis (% Under-seg, % FN, % Boundary, % FP)
    Applied across both Cut and Anhgoc imagery.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Study 3 Framework on {device} ===")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    if models_to_run is None:
        models_to_run = list(STUDY3_MODEL_CONFIGS.keys())

    all_results = {}
    timestamp = int(time.time())

    for src in sources:
        print(f"\n=======================================================")
        print(f"  PROCESSING DATA SOURCE: {src.upper()} IMAGERY")
        print(f"=======================================================")
        train_loader, val_loader, train_scenes, val_scenes = get_dataloaders(source=src)
        print(f"Train scenes: {len(train_scenes)} | Val scenes: {len(val_scenes)}")

        src_results = {}

        for m_key in models_to_run:
            m_name = STUDY3_MODEL_CONFIGS[m_key]["display_name"]
            print(f"\n--- [Model: {m_name}] on {src.upper()} ---")

            # Phase 1: Zero-shot / Pre-trained Evaluation
            print(f"  [Phase 1] Evaluating Pre-trained / Zero-shot Baseline...")
            model = build_study3_model(m_key, in_channels=9).to(device)
            pre_metrics, pre_errors, pre_visuals = evaluate_study3_model(model, val_loader, device=device)
            print(f"    Pretrained Val IoU: {pre_metrics['iou']:.4f} | Dice (F1): {pre_metrics['dice']:.4f} | Under-seg Err: {pre_errors['pct_under']:.1f}%")

            # Phase 2: Localized Fine-Tuning
            print(f"  [Phase 2] Running Localized Fine-Tuning ({num_epochs} epochs) with L_sem...")
            start_ft = time.time()
            model, history = finetune_study3_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=num_epochs,
                lr=lr
            )
            ft_duration = time.time() - start_ft

            # Post-adaptation Evaluation
            post_metrics, post_errors, post_visuals = evaluate_study3_model(model, val_loader, device=device)
            delta_iou = post_metrics["iou"] - pre_metrics["iou"]
            delta_dice = post_metrics["dice"] - pre_metrics["dice"]
            print(f"    Adapted Val IoU: {post_metrics['iou']:.4f} (+{delta_iou*100:.1f}pp) | Dice: {post_metrics['dice']:.4f} (+{delta_dice*100:.1f}pp)")
            print(f"    Under-seg Err: {pre_errors['pct_under']:.1f}% -> {post_errors['pct_under']:.1f}%")

            # Save checkpoint
            ckpt_path = os.path.join(MODELS_DIR, f"study3_{src}_{m_key}.pth")
            torch.save(model.state_dict(), ckpt_path)

            # Phase 3: Spatial Error Shift Plot
            plot_name = f"study3_error_shift_{src}_{m_key}.png"
            plot_path = os.path.join(PLOTS_DIR, plot_name)
            plot_spatial_error_shift(
                pre_errors,
                post_errors,
                save_path=plot_path,
                title_suffix=f"{m_name} ({src.upper()})"
            )

            # Visual prediction sample
            if post_visuals:
                vis_path = os.path.join(PLOTS_DIR, f"study3_pred_{src}_{m_key}.png")
                plot_prediction_spatial_errors(
                    post_visuals[0],
                    vis_path,
                    title=f"Study 3 Detection & Spatial Errors: {m_name} ({src.upper()})"
                )

            src_results[m_key] = {
                "display_name": m_name,
                "pretrained": {
                    "metrics": pre_metrics,
                    "spatial_errors": pre_errors
                },
                "finetuned": {
                    "metrics": post_metrics,
                    "spatial_errors": post_errors
                },
                "delta_ft": {
                    "iou_pp": delta_iou * 100.0,
                    "dice_pp": delta_dice * 100.0
                },
                "ft_duration_sec": ft_duration,
                "history": history
            }

        all_results[src] = src_results

    # Generate overarching comparison charts
    benchmark_chart_path = os.path.join(PLOTS_DIR, "study3_cut_vs_anhgoc_benchmark.png")
    plot_cut_vs_anhgoc_study3(all_results, benchmark_chart_path)

    # Replicate Figure 4 (Stacked bar chart for representative best model)
    best_model_key = "segformer_mitb5"
    if best_model_key in all_results.get("cut", {}):
        fig4_path = os.path.join(PLOTS_DIR, "study3_figure4_spatial_error_shift.png")
        plot_spatial_error_shift(
            all_results["cut"][best_model_key]["pretrained"]["spatial_errors"],
            all_results["cut"][best_model_key]["finetuned"]["spatial_errors"],
            save_path=fig4_path,
            title_suffix="SegFormer (MiT-B5) - Representative Architecture"
        )

    # Save complete JSON results
    results_json_path = os.path.join(RESULTS_DIR, "study3_domain_gap_results.json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll Study 3 results successfully saved to: {results_json_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study 3 Geographic Domain Gap & Localized Adaptation Runner")
    parser.add_argument("--sources", nargs="+", default=["cut", "anhgoc"], help="Data sources: cut, anhgoc")
    parser.add_argument("--models", nargs="+", default=list(STUDY3_MODEL_CONFIGS.keys()), help="Model keys")
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs per model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    run_study3_pipeline(
        sources=args.sources,
        models_to_run=args.models,
        num_epochs=args.epochs,
        lr=args.lr
    )
