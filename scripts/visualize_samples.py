import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch

from src.config import STUDY_AREA_BOUNDS
from src.dataset import discover_scenes, CanGioDataset
from src.evaluate import stretch_rgb

def create_sample_grid(output_path: str, source: str = "anhgoc"):
    scenes = discover_scenes("data", source=source)
    dataset = CanGioDataset(scenes, require_gt=True, augment=False, study_bounds=STUDY_AREA_BOUNDS)
    
    # Pick 4 representative scenes across different years
    target_dates = ["20061226", "20080825", "20110207", "20140215"]
    selected_indices = []
    
    for target in target_dates:
        for idx, scene in enumerate(dataset.scenes):
            if target in scene["date"]:
                selected_indices.append(idx)
                break
                
    if len(selected_indices) < 4:
        # Fallback to first 4
        selected_indices = list(range(min(4, len(dataset))))

    fig, axes = plt.subplots(len(selected_indices), 5, figsize=(20, 4.2 * len(selected_indices)))
    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    
    col_titles = [
        "Input: True Color (RGB)",
        "Input: False Color (NIR-R-G)",
        "Input: NDVI (Vegetation)",
        "Ground Truth: Disturbance Mask",
        "Ground Truth Overlay"
    ]
    
    for row_idx, data_idx in enumerate(selected_indices):
        image_tensor, mask_tensor, meta = dataset[data_idx]
        img_np = image_tensor.numpy()
        mask_np = mask_tensor.squeeze().numpy()
        
        # Channels: 0: Blue, 1: Green, 2: Red, 3: NIR
        blue = img_np[0]
        green = img_np[1]
        red = img_np[2]
        nir = img_np[3]
        
        # 1. RGB
        rgb = np.stack([red, green, blue], axis=-1)
        rgb_disp = stretch_rgb(rgb)
        
        # 2. False Color (NIR, Red, Green)
        cir = np.stack([nir, red, green], axis=-1)
        cir_disp = stretch_rgb(cir)
        
        # 3. NDVI
        ndvi = (nir - red) / (nir + red + 1e-7)
        
        # 4. Ground Truth Mask
        gt_binary = (mask_np > 0.5).astype(np.float32)
        pos_pixels = int(gt_binary.sum())
        
        # 5. Overlay on dimmed True Color
        dimmed_rgb = rgb_disp * 0.65
        overlay = dimmed_rgb.copy()
        # Highlight disturbance pixels in bright neon yellow/red
        overlay[gt_binary == 1] = [1.0, 0.2, 0.1]
        
        # Plot columns
        ax_rgb = axes[row_idx, 0]
        ax_rgb.imshow(rgb_disp)
        ax_rgb.set_ylabel(f"{meta['sensor']}\n{meta['date']}\n({meta['year_folder']})", fontsize=12, fontweight='bold')
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])
        if row_idx == 0:
            ax_rgb.set_title(col_titles[0], fontsize=13, fontweight='bold', pad=10)
            
        ax_cir = axes[row_idx, 1]
        ax_cir.imshow(cir_disp)
        ax_cir.set_xticks([])
        ax_cir.set_yticks([])
        if row_idx == 0:
            ax_cir.set_title(col_titles[1], fontsize=13, fontweight='bold', pad=10)
            
        ax_ndvi = axes[row_idx, 2]
        im_ndvi = ax_ndvi.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        ax_ndvi.set_xticks([])
        ax_ndvi.set_yticks([])
        if row_idx == 0:
            ax_ndvi.set_title(col_titles[2], fontsize=13, fontweight='bold', pad=10)
            
        ax_gt = axes[row_idx, 3]
        cmap_gt = ListedColormap(["#f5f5f5", "#b30000"])
        ax_gt.imshow(gt_binary, cmap=cmap_gt, vmin=0, vmax=1)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        ax_gt.set_xlabel(f"Disturbed: {pos_pixels} px", fontsize=10, fontweight='bold', color="#800000")
        if row_idx == 0:
            ax_gt.set_title(col_titles[3], fontsize=13, fontweight='bold', pad=10)
            
        ax_ov = axes[row_idx, 4]
        ax_ov.imshow(overlay)
        ax_ov.set_xticks([])
        ax_ov.set_yticks([])
        if row_idx == 0:
            ax_ov.set_title(col_titles[4], fontsize=13, fontweight='bold', pad=10)
            
    # Add main title
    fig.suptitle("Can Gio Mangrove Forest Disturbance Dataset Samples & Ground Truth\n(Cần Giờ Biosphere Reserve, Vietnam: 2006–2014)", 
                 fontsize=16, fontweight='bold', y=0.99)
                 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Sample grid saved successfully to {output_path}")

if __name__ == "__main__":
    artifact_dir = r"C:\Users\hai49\.gemini\antigravity\brain\1d70dc2a-5607-46e1-93b6-d468e1db4e11"
    output_img = os.path.join(artifact_dir, "dataset_samples_overview.png")
    create_sample_grid(output_img)
    # Also save a copy in results/plots/
    create_sample_grid("results/plots/dataset_samples_overview.png")
