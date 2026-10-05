import os
import sys
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DATA_DIR, PREDICTIONS_DIR, PLOTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.evaluate import stretch_rgb, compute_binary_metrics, compute_boundary_iou


def generate_waveattn_sopseg_eval_figure(
    scene_id: str = "LC08_20140810",
    output_path: str = "results/plots/eval_waveattn_sopseg_full_LC08_20140810.png",
    threshold: float = 0.5
):
    """
    Plots a comprehensive 6-panel evaluation figure for WaveAttn SOPSeg on Full Radiometric Scenes,
    following the structure of eval_LC08_20140810.png:
    1. True Color (RGB)
    2. False Color (NIR-R-G)
    3. Vegetation Index (NDVI)
    4. Ground Truth Mask (Positives: N)
    5. Predicted Disturbance Probability
    6. Evaluation Map (Green: TP, Red: FP, Blue: FN)
    """
    # 1. Load scene data from Full Scenes (anhgoc source)
    scenes = discover_scenes(DATA_DIR, source="anhgoc")
    sc = next((s for s in scenes if scene_id in s["scene_id"]), None)
    if sc is None:
        raise ValueError(f"Scene {scene_id} not found in full scenes dataset.")
        
    ds = CanGioDataset([sc], require_gt=True)
    img_tensor, gt_tensor, meta = ds[0]
    img_np = img_tensor.numpy()
    gt = (gt_tensor.numpy().squeeze() > 0.5).astype(np.float32)

    # 2. Load WaveAttn SOPSeg predictions (Full Scenes)
    pred_path = os.path.join(PREDICTIONS_DIR, f"sopseg_opt1_wave_anhgoc_{scene_id}.tif")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    with rasterio.open(pred_path) as src:
        pred_bin = src.read(1).astype(np.float32)
        pred_prob = src.read(2).astype(np.float32)

    # 3. Calculate metrics
    m = compute_binary_metrics(pred_bin, gt, threshold=threshold)
    biou = compute_boundary_iou(pred_bin, gt, threshold=threshold)

    # 4. Extract Bands: 0: Blue, 1: Green, 2: Red, 3: NIR
    blue = img_np[0]
    green = img_np[1]
    red = img_np[2]
    nir = img_np[3] if img_np.shape[0] > 3 else red

    # RGB composite (Red, Green, Blue)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_disp = stretch_rgb(rgb)

    # False Color composite (NIR, Red, Green)
    false_color = np.stack([nir, red, green], axis=-1)
    false_color_disp = stretch_rgb(false_color)

    # NDVI map
    ndvi = (nir - red) / (nir + red + 1e-7)

    # 5. Build Confusion Overlay on RGB
    h, w = gt.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    tp = (pred_bin == 1) & (gt == 1)
    fp = (pred_bin == 1) & (gt == 0)
    fn = (pred_bin == 0) & (gt == 1)

    # Vivid colors for maximum clarity
    overlay[tp] = [0.0, 0.90, 0.20]  # Green: Correct detection (TP)
    overlay[fp] = [0.95, 0.15, 0.15]  # Red: False alarm (FP)
    overlay[fn] = [0.15, 0.45, 0.95]  # Blue: Missed (FN)

    # Base background: dimmed RGB
    eval_map = rgb_disp * 0.4 + overlay * 0.6
    # Where mask exists, enhance contrast
    mask_active = (tp | fp | fn)
    eval_map[mask_active] = overlay[mask_active]

    # 6. Plotting 2x3 Grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), dpi=300)
    
    # Title with metrics
    title_text = (
        f"Can Gio Disturbance Segmentation \u2014 WaveAttn SOPSeg ({scene_id})\n"
        f"Full Radiometric Scene | IoU: {m['iou']:.4f} | Boundary IoU: {biou:.4f} | "
        f"Dice: {m['dice']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f}"
    )
    fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.98)

    # Top-Left: True Color
    axes[0, 0].imshow(rgb_disp)
    axes[0, 0].set_title("True Color (RGB)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    # Top-Middle: False Color
    axes[0, 1].imshow(false_color_disp)
    axes[0, 1].set_title("False Color (NIR-R-G)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 1].axis("off")

    # Top-Right: NDVI
    im_ndvi = axes[0, 2].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0, 2].set_title("Vegetation Index (NDVI)", fontsize=12, fontweight="bold", pad=8)
    axes[0, 2].axis("off")
    cbar_ndvi = fig.colorbar(im_ndvi, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar_ndvi.ax.tick_params(labelsize=9)

    # Bottom-Left: Ground Truth
    im_gt = axes[1, 0].imshow(gt, cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Ground Truth (Positives: {int(gt.sum())} px)", fontsize=12, fontweight="bold", pad=8)
    axes[1, 0].axis("off")

    # Bottom-Middle: Predicted Probability
    im_pred = axes[1, 1].imshow(pred_prob, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Predicted Probability (Threshold: {threshold})", fontsize=12, fontweight="bold", pad=8)
    axes[1, 1].axis("off")
    cbar_pred = fig.colorbar(im_pred, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar_pred.ax.tick_params(labelsize=9)

    # Bottom-Right: Evaluation Map
    axes[1, 2].imshow(eval_map)
    axes[1, 2].set_title(
        f"Evaluation Map (TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']})",
        fontsize=12, fontweight="bold", pad=8
    )
    axes[1, 2].axis("off")

    # Legend for Evaluation Map at bottom
    legend_patches = [
        mpatches.Patch(color=[0.0, 0.90, 0.20], label=f"True Positive (TP: {m['tp']} px)"),
        mpatches.Patch(color=[0.95, 0.15, 0.15], label=f"False Positive (FP: {m['fp']} px)"),
        mpatches.Patch(color=[0.15, 0.45, 0.95], label=f"False Negative (FN: {m['fn']} px)")
    ]
    axes[1, 2].legend(
        handles=legend_patches, loc="lower right", fontsize=8.5,
        framealpha=0.85, facecolor="#FFFFFF", edgecolor="#D1D5DB"
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated WaveAttn SOPSeg evaluation figure at: {output_path}")


if __name__ == "__main__":
    generate_waveattn_sopseg_eval_figure(
        scene_id="LC08_20140810",
        output_path="results/plots/eval_waveattn_sopseg_full_LC08_20140810.png"
    )
