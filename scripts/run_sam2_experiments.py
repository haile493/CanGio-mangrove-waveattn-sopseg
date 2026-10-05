import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import BiTemporalCanGioDataset, discover_scenes
from src.evaluate import compute_binary_metrics, stretch_rgb
from src.sam2_cd import SAM2CDWrapper, train_sam2_cd, evaluate_sam2_cd
from src.bisam_cd import BiSAMCDPredictor, evaluate_bisam_cd


def plot_change_detection_sample(
    t1_tensor: torch.Tensor,
    t2_tensor: torch.Tensor,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    method_name: str,
    data_source: str,
    meta: dict,
    save_path: str,
    threshold: float = 0.5
):
    """
    Renders 6-panel visualization:
    1. T1 Pre-image RGB
    2. T2 Post-image RGB
    3. Spectral Delta NDVI (T2 - T1)
    4. Ground Truth Mask
    5. Predicted Change Probability
    6. Classification Confusion (TP: Green, FP: Red, FN: Blue)
    """
    t1 = t1_tensor.detach().cpu().numpy()
    t2 = t2_tensor.detach().cpu().numpy()

    # Red=2, Green=1, Blue=0
    rgb1 = stretch_rgb(np.stack([t1[2], t1[1], t1[0]], axis=-1))
    rgb2 = stretch_rgb(np.stack([t2[2], t2[1], t2[0]], axis=-1))

    # Delta NDVI if red and nir exist
    if t1.shape[0] >= 4:
        nir1, red1 = t1[3], t1[2]
        nir2, red2 = t2[3], t2[2]
    else:
        # Fallback to green/red contrast
        nir1, red1 = t1[1], t1[2]
        nir2, red2 = t2[1], t2[2]

    ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-7)
    ndvi2 = (nir2 - red2) / (nir2 + red2 + 1e-7)
    delta_ndvi = ndvi2 - ndvi1

    gt = (gt_mask >= 0.5).astype(np.float32)
    bin_pred = (pred_prob >= threshold).astype(np.float32)

    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    tp = (bin_pred == 1) & (gt == 1)
    fp = (bin_pred == 1) & (gt == 0)
    fn = (bin_pred == 0) & (gt == 1)

    overlay[tp] = [0.0, 0.9, 0.0]  # Green: True Positive
    overlay[fp] = [0.9, 0.1, 0.1]  # Red: False Positive
    overlay[fn] = [0.1, 0.4, 0.9]  # Blue: False Negative

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    t1_date = meta.get("t1_date", "T1")
    t2_date = meta.get("t2_date", "T2")
    fig.suptitle(
        f"{method_name} ({data_source.upper()} Imagery) | Can Gio Mangrove\nT1 ({t1_date}) -> T2 ({t2_date})",
        fontsize=14, fontweight="bold"
    )

    axes[0, 0].imshow(rgb1)
    axes[0, 0].set_title(f"Pre-image RGB ({t1_date})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb2)
    axes[0, 1].set_title(f"Post-image RGB ({t2_date})")
    axes[0, 1].axis("off")

    im_diff = axes[0, 2].imshow(delta_ndvi, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title(r"Spectral $\Delta$NDVI ($T_2 - T_1$)")
    axes[0, 2].axis("off")
    fig.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(gt, cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Ground Truth Disturbance (Pos: {int(gt.sum())})")
    axes[1, 0].axis("off")

    im_pred = axes[1, 1].imshow(pred_prob, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Predicted Probability (Thresh: {threshold})")
    axes[1, 1].axis("off")
    fig.colorbar(im_pred, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Confusion (Green: TP, Red: FP, Blue: FN)")
    axes[1, 2].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved visualization: {save_path}")


def generate_benchmark_bar_chart(benchmark_data: Dict[str, Dict[str, Dict[str, float]]], output_path: str):
    """
    Renders comparative 4-metric bar chart comparing SAM2-CD and BiSAM-CD across Cut vs Anhgoc.
    """
    metrics_list = ["iou", "dice", "precision", "recall"]
    metric_titles = ["IoU", "Dice (F1)", "Precision", "Recall"]

    models = ["SAM2-CD (Supervised)", "BiSAM-CD (Zero-Shot)"]
    model_keys = ["sam2_cd", "bisam_cd"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    x = np.arange(len(models))
    width = 0.35

    color_cut = "#1f77b4"     # Blue
    color_anhgoc = "#ff7f0e"  # Orange

    for i, (m_key, m_title) in enumerate(zip(metrics_list, metric_titles)):
        ax = axes[i]
        vals_cut = [benchmark_data["cut"][mk][m_key] for mk in model_keys]
        vals_anhgoc = [benchmark_data["anhgoc"][mk][m_key] for mk in model_keys]

        rects1 = ax.bar(x - width/2, vals_cut, width, label="Cut Chips", color=color_cut, alpha=0.88, edgecolor="black")
        rects2 = ax.bar(x + width/2, vals_anhgoc, width, label="Anhgoc (Full)", color=color_anhgoc, alpha=0.88, edgecolor="black")

        ax.set_title(m_title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#b34700", fontweight="bold")

        if i == 0:
            ax.legend(frameon=True, loc="upper right")

    fig.suptitle("Comparison of SAM2-CD and BiSAM-CD on Can Gio Satellite Imagery (Cut vs. Anhgoc)",
                 fontsize=14, fontweight="bold", y=1.03)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Benchmark chart saved to: {output_path}")


def run_experiments(
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 5e-5,
    in_channels: int = 9,
    force_retrain: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 80)
    print(" SAM2 REMOTE SENSING CHANGE DETECTION BENCHMARK")
    print(f" Models: [1] SAM2-CD (JSTARS 2025) | [2] BiSAM-CD (TGRS 2025)")
    print(f" Datasets: Cut (pre-cut chips) vs Anhgoc (original scenes)")
    print(f" Channels: {in_channels} | Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    val_years = ["2011", "2014"]
    benchmark_results = {"cut": {}, "anhgoc": {}}

    # Initialize BiSAM-CD predictor once
    bisam_predictor = BiSAMCDPredictor(device=device)

    spectral_bands = ["blue", "green", "red"] if in_channels == 3 else ["blue", "green", "red", "nir", "swir1", "swir2"]
    include_indices = (in_channels != 3)

    for source in ["cut", "anhgoc"]:
        print(f"\n#################################################################")
        print(f"  PROCESSING DATA SOURCE: {source.upper()}")
        print(f"#################################################################")

        scenes = discover_scenes(os.path.join(PROJECT_ROOT, "data"), source=source)
        val_scenes = [s for s in scenes if any(y in s["year_folder"] for y in val_years) and s.get("groundtruth_path")]

        # BiTemporal dataset for validation (holdout years 2011 & 2014)
        val_ds = BiTemporalCanGioDataset(
            val_scenes,
            pairing_strategy="all",
            spectral_bands=spectral_bands,
            include_indices=include_indices,
            augment=False
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        print(f"Validation dataset size ({source}): {len(val_ds)} bitemporal pairs")

        # -----------------------------------------------------------------
        # 1. SAM2-CD: Supervised fine-tuning with ASG + MSFFM + WF + GLCAM
        # -----------------------------------------------------------------
        ckpt_path = os.path.join(MODELS_DIR, f"best_sam2_cd_{source}.pth")
        if os.path.exists(ckpt_path) and not force_retrain:
            print(f"\n[1/2] Loading existing SAM2-CD checkpoint from {ckpt_path}...")
            sam2_cd_model = SAM2CDWrapper(in_channels=in_channels, device=device).to(device)
            sam2_cd_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            sam2_cd_metrics = evaluate_sam2_cd(sam2_cd_model, val_loader, device=device)
        else:
            print(f"\n[1/2] Training SAM2-CD on {source.upper()} for {epochs} epochs...")
            sam2_cd_model, sam2_cd_metrics = train_sam2_cd(
                source=source,
                in_channels=in_channels,
                num_epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                save_ckpt=True
            )

        benchmark_results[source]["sam2_cd"] = sam2_cd_metrics
        print(f"SAM2-CD ({source.upper()}) Validation Metrics: {sam2_cd_metrics}")

        # -----------------------------------------------------------------
        # 2. BiSAM-CD: Zero-Shot Bidirectional Temporal Memory Tracking
        # -----------------------------------------------------------------
        print(f"\n[2/2] Running BiSAM-CD (Zero-Shot) on {source.upper()} validation split...")
        bisam_metrics = evaluate_bisam_cd(bisam_predictor, val_loader, max_samples=len(val_ds))
        benchmark_results[source]["bisam_cd"] = bisam_metrics
        print(f"BiSAM-CD ({source.upper()}) Validation Metrics: {bisam_metrics}")

        # Generate sample visual comparison on first validation pair
        t1, t2, gt_mask, meta = val_ds[0]
        with torch.no_grad():
            p2, _, _ = sam2_cd_model(t1.unsqueeze(0).to(device).float(), t2.unsqueeze(0).to(device).float())
            sam2_cd_pred = torch.sigmoid(p2)[0, 0].cpu().numpy()

        bisam_pred = bisam_predictor.predict_change_map(t1, t2)

        # Plot SAM2-CD sample
        plot_change_detection_sample(
            t1, t2, gt_mask[0].numpy(), sam2_cd_pred,
            method_name="SAM2-CD (Supervised)",
            data_source=source,
            meta=meta,
            save_path=os.path.join(PLOTS_DIR, f"sam2_cd_{source}_sample.png")
        )

        # Plot BiSAM-CD sample
        plot_change_detection_sample(
            t1, t2, gt_mask[0].numpy(), bisam_pred,
            method_name="BiSAM-CD (Zero-Shot)",
            data_source=source,
            meta=meta,
            save_path=os.path.join(PLOTS_DIR, f"bisam_cd_{source}_sample.png")
        )

    # Save complete benchmark metrics JSON
    metrics_json_path = os.path.join(RESULTS_DIR, "sam2_papers_benchmark.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nFull benchmark results saved to: {metrics_json_path}")

    # Generate comparative bar chart
    chart_path = os.path.join(PLOTS_DIR, "sam2_papers_cut_vs_anhgoc_comparison.png")
    generate_benchmark_bar_chart(benchmark_results, chart_path)

    # Print summary table
    print("\n" + "=" * 90)
    print("                   FINAL BENCHMARK SUMMARY (HOLDOUT 2011 & 2014)")
    print("=" * 90)
    print(f"{'Method':<25} | {'Data Source':<10} | {'IoU':<8} | {'Dice (F1)':<10} | {'Precision':<10} | {'Recall':<10} | {'Accuracy':<10}")
    print("-" * 90)
    for src in ["cut", "anhgoc"]:
        for m_key, m_name in [("sam2_cd", "SAM2-CD (Supervised)"), ("bisam_cd", "BiSAM-CD (Zero-Shot)")]:
            m = benchmark_results[src][m_key]
            print(f"{m_name:<25} | {src.upper():<10} | {m['iou']:<8.4f} | {m['dice']:<10.4f} | {m['precision']*100:<9.2f}% | {m['recall']*100:<9.2f}% | {m['accuracy']*100:<9.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM2-CD and BiSAM-CD on Cut and Anhgoc imagery")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs for SAM2-CD")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--in_channels", type=int, default=9, help="Number of input channels (3 for RGB, 9 for all bands+indices)")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain models even if checkpoint exists")
    args = parser.parse_args()

    run_experiments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        in_channels=args.in_channels,
        force_retrain=args.force_retrain
    )
