import os
import sys
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DATA_DIR, PREDICTIONS_DIR, PLOTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.evaluate import stretch_rgb, compute_binary_metrics, compute_boundary_iou


def load_scene_data(source: str = "cut"):
    scenes = discover_scenes(DATA_DIR, source=source)
    sc = next((s for s in scenes if "LT05_20110207" in s["scene_id"]), scenes[0])
    ds = CanGioDataset([sc])
    img_t, gt_t, meta = ds[0]
    img = img_t.numpy()
    gt = (gt_t.numpy().squeeze() > 0.5).astype(np.uint8)
    
    # NIR-R-G composite
    # Band index 3: NIR, Band index 2: Red, Band index 1: Green
    nir = img[3]
    red = img[2]
    green = img[1]
    cir = stretch_rgb(np.stack([nir, red, green], axis=-1))
    
    return cir, gt, meta


def generate_figure4(output_path: str, style: str = "error_coded"):
    """
    Generates Figure 4 comparing Cut Chips vs Full Radiometric Scenes
    across the 6 specified panels:
    (a) False-Color NIR-R-G
    (b) Ground Truth Disturbance
    (c) Fully-Supervised U-Net
    (d) SAM + LoRA
    (e) Standard SOPSeg
    (f) WaveAttn SOPSeg
    """
    cir_cut, gt_cut, _ = load_scene_data("cut")
    cir_full, gt_full, _ = load_scene_data("anhgoc")
    
    # Load model predictions
    unet_cut = np.load(os.path.join(PREDICTIONS_DIR, "unet_resnet34_cut_LT05_20110207.npy"))
    unet_full = np.load(os.path.join(PREDICTIONS_DIR, "unet_resnet34_anhgoc_LT05_20110207.npy"))
    
    sam_cut = np.load(os.path.join(PREDICTIONS_DIR, "sam_lora_cut_LT05_20110207.npy"))
    sam_full = np.load(os.path.join(PREDICTIONS_DIR, "sam_lora_anhgoc_LT05_20110207.npy"))
    
    with rasterio.open(os.path.join(PREDICTIONS_DIR, "sopseg_cut_LT05_20110207_pred.tif")) as f:
        sop_cut = f.read(1).astype(np.uint8)
    with rasterio.open(os.path.join(PREDICTIONS_DIR, "sopseg_anhgoc_LT05_20110207_pred.tif")) as f:
        sop_full = f.read(1).astype(np.uint8)
        
    with rasterio.open(os.path.join(PREDICTIONS_DIR, "sopseg_opt1_wave_cut_LT05_20110207.tif")) as f:
        wave_cut = f.read(1).astype(np.uint8)
    with rasterio.open(os.path.join(PREDICTIONS_DIR, "sopseg_opt1_wave_anhgoc_LT05_20110207.tif")) as f:
        wave_full = f.read(1).astype(np.uint8)
        
    fig, axes = plt.subplots(2, 6, figsize=(18, 7.2), dpi=300)
    plt.subplots_adjust(wspace=0.08, hspace=0.28, left=0.08, right=0.98, top=0.91, bottom=0.13)
    
    col_titles = [
        "(a) False-Color (NIR-R-G)",
        "(b) Ground Truth",
        "(c) U-Net (ResNet-34)",
        "(d) SAM + LoRA",
        "(e) Standard SOPSeg",
        "(f) WaveAttn SOPSeg"
    ]
    
    rows_data = [
        ("Cut Chips\n(124 \u00d7 124)", cir_cut, gt_cut, unet_cut, sam_cut, sop_cut, wave_cut),
        ("Full Radiometric\nScenes", cir_full, gt_full, unet_full, sam_full, sop_full, wave_full)
    ]
    
    bg_color = np.array([0.06, 0.09, 0.16]) # dark navy slate #0F172A
    tp_color = np.array([0.13, 0.77, 0.37]) # vivid emerald green #22C55E
    fp_color = np.array([0.94, 0.27, 0.27]) # vivid red-coral #EF4444
    fn_color = np.array([0.23, 0.51, 0.96]) # vivid blue #3B82F6
    gt_color = np.array([0.98, 0.75, 0.14]) # vivid amber gold #FBBF24
    
    for row_idx, (row_label, cir, gt, unet, sam, sop, wave) in enumerate(rows_data):
        # 1. False Color
        axes[row_idx, 0].imshow(cir)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, fontweight="bold", labelpad=12, rotation=0, va="center", ha="right")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        for spine in axes[row_idx, 0].spines.values():
            spine.set_edgecolor("#9CA3AF")
            spine.set_linewidth(1.0)
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=11.5, fontweight="bold", pad=9)
        axes[row_idx, 0].text(0.5, -0.09, "LT05_20110207", transform=axes[row_idx, 0].transAxes,
                              ha="center", va="top", fontsize=9, color="#4B5563")

        # 2. Ground Truth
        gt_rgb = np.tile(bg_color, (gt.shape[0], gt.shape[1], 1))
        gt_rgb[gt == 1] = gt_color
        axes[row_idx, 1].imshow(gt_rgb)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])
        for spine in axes[row_idx, 1].spines.values():
            spine.set_edgecolor("#9CA3AF")
            spine.set_linewidth(1.0)
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_titles[1], fontsize=11.5, fontweight="bold", pad=9)
        axes[row_idx, 1].text(0.5, -0.09, f"Positives: {int(gt.sum())} px", transform=axes[row_idx, 1].transAxes,
                              ha="center", va="top", fontsize=9, color="#4B5563")

        # 3-6. Models
        models = [unet, sam, sop, wave]
        for col_idx, pred in enumerate(models, start=2):
            m = compute_binary_metrics(pred, gt)
            biou = compute_boundary_iou(pred, gt)
            
            if style == "error_coded":
                pred_rgb = np.tile(bg_color, (pred.shape[0], pred.shape[1], 1))
                tp = np.logical_and(pred == 1, gt == 1)
                fp = np.logical_and(pred == 1, gt == 0)
                fn = np.logical_and(pred == 0, gt == 1)
                
                pred_rgb[tp] = tp_color
                pred_rgb[fp] = fp_color
                pred_rgb[fn] = fn_color
                axes[row_idx, col_idx].imshow(pred_rgb)
            else: # binary style
                pred_rgb = np.tile(bg_color, (pred.shape[0], pred.shape[1], 1))
                pred_rgb[pred == 1] = gt_color
                axes[row_idx, col_idx].imshow(pred_rgb)
                
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_edgecolor("#9CA3AF")
                spine.set_linewidth(1.0)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=11.5, fontweight="bold", pad=9)
                
            metric_txt = f"IoU: {m['iou']:.3f} | BIoU: {biou:.3f}"
            axes[row_idx, col_idx].text(0.5, -0.09, metric_txt, transform=axes[row_idx, col_idx].transAxes,
                                        ha="center", va="top", fontsize=8.8, fontweight="semibold", color="#111827")

    # Add legend at bottom
    if style == "error_coded":
        legend_elements = [
            mpatches.Patch(color=gt_color, label="Ground Truth Disturbance"),
            mpatches.Patch(color=tp_color, label="True Positive (TP)"),
            mpatches.Patch(color=fp_color, label="False Positive (FP - Over-prediction)"),
            mpatches.Patch(color=fn_color, label="False Negative (FN - Missed Disturbance)")
        ]
        fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.54, 0.02),
                   ncol=4, frameon=True, fontsize=10.5, facecolor="#F9FAFB", edgecolor="#D1D5DB")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4 ({style}) to {output_path}")


if __name__ == "__main__":
    generate_figure4("results/plots/figure4_qualitative_predictions_cut_vs_full.png", style="error_coded")
    generate_figure4("results/plots/figure4_qualitative_predictions_binary.png", style="binary")
    # Also overwrite the original path used by export script
    generate_figure4("results/plots/sopseg_integrated_visual_comparison.png", style="error_coded")
