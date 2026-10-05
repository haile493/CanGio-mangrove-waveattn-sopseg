"""
Figure 5 Reproduction Script
Component Ablation Study on Full Radiometric Scenes
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def generate_waveattn_ablation_plot(output_path: str = "results/plots/sopseg_ablation_study_full.png"):
    # Ablation stages on Full Radiometric Scenes
    stages = [
        "Base\n(SAM FT)",
        "+RAM\n(Box Prompt)",
        "+RAM + OPM\n(Oriented)",
        "SOPSeg\n(Frozen ViT)",
        "SOPSeg\n+ LoRA",
        "WaveAttn SOPSeg\n(+WaveFusion+NBR)"
    ]
    
    ious = [0.5413, 0.1412, 0.5842, 0.7378, 0.7474, 0.7576]
    bious = [0.4180, 0.6240, 0.8872, 0.9641, 0.9654, 0.9634]
    dices = [0.7024, 0.2475, 0.7376, 0.8492, 0.8554, 0.8621]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.2, 1]})
    
    x = np.arange(len(stages))
    width = 0.35
    
    # Left subplot: Bar comparison of IoU vs Boundary IoU
    r_iou = ax1.bar(x - width/2, ious, width, label="Mask IoU", color="#1f77b4", alpha=0.88, edgecolor="black", linewidth=0.8)
    r_biou = ax1.bar(x + width/2, bious, width, label="Boundary IoU (BIoU)", color="#9467bd", alpha=0.88, edgecolor="black", linewidth=0.8)
    
    ax1.set_title("Ablation Progression on Full Radiometric Scenes (IoU vs BIoU)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.12)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, loc="upper left", fontsize=9.5)
    
    # Annotate numbers
    for r in r_iou:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", (r.get_x() + r.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#0b4f8a")
    for r in r_biou:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}", (r.get_x() + r.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#5c2d91")
        
    # Right subplot: Overall trend line across stages
    ax2.plot(x, ious, marker='o', linewidth=2.5, markersize=8, color="#1f77b4", label="Mask IoU")
    ax2.plot(x, dices, marker='s', linewidth=2.5, markersize=8, color="#2ca02c", label="Dice / F1")
    ax2.plot(x, bious, marker='^', linewidth=2.5, markersize=8, color="#9467bd", label="Boundary IoU (BIoU)")
    
    ax2.set_title("Performance Trajectory Across Stages", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, loc="lower right", fontsize=9.5)
    
    # Annotate final peak on right subplot
    ax2.annotate(f"Peak IoU: 0.758\nPeak Dice: 0.862", 
                 xy=(5, 0.7576), xytext=(3.6, 0.60),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#eef2f6", ec="#1B365D"))
    
    fig.suptitle("Component Ablation Study on Full Radiometric Scenes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Full Scene ablation plot successfully generated at: {output_path}")

# Backward compatibility alias
generate_anhgoc_ablation_plot = generate_waveattn_ablation_plot

if __name__ == "__main__":
    out_p = "results/plots/sopseg_ablation_study_full.png"
    generate_waveattn_ablation_plot(out_p)
