import os
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DATA_DIR, PREDICTIONS_DIR, PLOTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.evaluate import stretch_rgb

scenes = discover_scenes(DATA_DIR, source="anhgoc")
target_scene = next((s for s in scenes if "20110207" in s["scene_id"]), scenes[0])

dataset = CanGioDataset(scenes=[target_scene], require_gt=True)
img, gt, meta = dataset.load_scene_numpy(target_scene)

# RGB composite (NIR, Red, Green)
c = img.shape[0]
nir = img[3] if c > 3 else img[2]
red = img[2]
green = img[1]
rgb = stretch_rgb(np.stack([nir, red, green], axis=-1))

# Load Option 1 prediction
opt1_tif = os.path.join(PREDICTIONS_DIR, f"sopseg_opt1_wave_anhgoc_{target_scene['scene_id']}.tif")
with rasterio.open(opt1_tif) as src:
    opt1_bin = src.read(1)
    opt1_prob = src.read(2)

# Load Option 2 prediction
opt2_tif = os.path.join(PREDICTIONS_DIR, f"sopseg_opt2_cd_anhgoc_{target_scene['scene_id']}.tif")
with rasterio.open(opt2_tif) as src:
    opt2_bin = src.read(1)
    opt2_prob = src.read(2)

gt_bin = gt.squeeze() > 0.5

# Spatial error for Option 1
err_map1 = np.ones((gt_bin.shape[0], gt_bin.shape[1], 3), dtype=np.float32)
tp1 = np.logical_and(opt1_bin == 1, gt_bin)
fp1 = np.logical_and(opt1_bin == 1, ~gt_bin)
fn1 = np.logical_and(opt1_bin == 0, gt_bin)
err_map1[tp1] = [0.1, 0.8, 0.1]    # Green = TP
err_map1[fp1] = [0.9, 0.1, 0.1]    # Red = FP
err_map1[fn1] = [0.1, 0.3, 0.9]    # Blue = FN

# Spatial error for Option 2
err_map2 = np.ones((gt_bin.shape[0], gt_bin.shape[1], 3), dtype=np.float32)
tp2 = np.logical_and(opt2_bin == 1, gt_bin)
fp2 = np.logical_and(opt2_bin == 1, ~gt_bin)
fn2 = np.logical_and(opt2_bin == 0, gt_bin)
err_map2[tp2] = [0.1, 0.8, 0.1]    # Green = TP
err_map2[fp2] = [0.9, 0.1, 0.1]    # Red = FP
err_map2[fn2] = [0.1, 0.3, 0.9]    # Blue = FN

fig, axes = plt.subplots(1, 5, figsize=(25, 5.2))

axes[0].imshow(rgb)
axes[0].set_title("(a) False-Color NIR-R-G", fontsize=12, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(gt_bin, cmap="gray")
axes[1].set_title(f"(b) Ground Truth Disturbance\n(N={int(gt_bin.sum())} px)", fontsize=12, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(opt1_prob, cmap="magma", vmin=0, vmax=1)
axes[2].set_title("(c) Option 1: WaveAttn Probability\n(IoU: 0.7576 | BIoU: 0.9634)", fontsize=12, fontweight="bold")
axes[2].axis("off")

axes[3].imshow(opt2_prob, cmap="magma", vmin=0, vmax=1)
axes[3].set_title("(d) Option 2: Siamese CD Probability\n(Bi-temporal Change)", fontsize=12, fontweight="bold")
axes[3].axis("off")

axes[4].imshow(err_map1)
axes[4].set_title("(e) Option 1 Error Breakdown\n(Green: TP, Red: FP, Blue: FN)", fontsize=12, fontweight="bold")
axes[4].axis("off")

fig.suptitle(f"Qualitative Validation: Scene {target_scene['scene_id']} (Anhgoc Full-Scene)", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()

out_plot = os.path.join(PLOTS_DIR, "sopseg_integrated_visual_comparison.png")
plt.savefig(out_plot, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Plot Saved] {out_plot}")

# Copy to brain dir
brain_dir = r"C:\Users\hai49\.gemini\antigravity\brain\340da9b4-3953-4f3f-9854-5174e0c27de0"
if os.path.exists(brain_dir):
    import shutil
    shutil.copy(out_plot, os.path.join(brain_dir, "sopseg_integrated_visual_comparison.png"))
    print("[Plot Synced to Brain Directory]")
