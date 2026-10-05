import os
import json
import matplotlib.pyplot as plt
import numpy as np

RESULTS_JSON = "results/sopseg_benchmark_results.json"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(RESULTS_JSON, "r") as f:
    results_summary = json.load(f)

# Benchmark Chart: Cut vs Anhgoc across 5 methods
fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
methods = ["U-Net", "Superpixel WSL", "SAM + LoRA", "SOPSeg (Frozen)", "SOPSeg + LoRA"]
metrics_names = ["iou", "biou", "dice", "precision"]
titles = ["IoU", "Boundary IoU (BIoU)", "Dice (F1)", "Precision"]

cut_vals = {
    "iou": [0.3850, 0.0196, 0.5732, 0.6980, results_summary["sources"]["cut"]["global_metrics"]["iou"]],
    "biou": [0.2810, 0.0120, 0.4410, 0.9619, results_summary["sources"]["cut"]["global_metrics"]["biou"]],
    "dice": [0.5559, 0.0384, 0.7287, 0.8222, results_summary["sources"]["cut"]["global_metrics"]["dice"]],
    "precision": [0.4299, 0.0196, 0.8916, 0.7363, results_summary["sources"]["cut"]["global_metrics"]["precision"]]
}

anhgoc_vals = {
    "iou": [0.3782, 0.0147, 0.5413, 0.7378, results_summary["sources"]["anhgoc"]["global_metrics"]["iou"]],
    "biou": [0.2740, 0.0090, 0.4180, 0.9641, results_summary["sources"]["anhgoc"]["global_metrics"]["biou"]],
    "dice": [0.5488, 0.0290, 0.7024, 0.8492, results_summary["sources"]["anhgoc"]["global_metrics"]["dice"]],
    "precision": [0.4236, 0.0147, 0.8980, 0.7755, results_summary["sources"]["anhgoc"]["global_metrics"]["precision"]]
}

x = np.arange(len(methods))
width = 0.35

for idx, (m_key, title) in enumerate(zip(metrics_names, titles)):
    ax = axes[idx]
    v_cut = cut_vals[m_key]
    v_anh = anhgoc_vals[m_key]

    r1 = ax.bar(x - width/2, v_cut, width, label="Cut Chips", color="#1f77b4", alpha=0.88, edgecolor="black")
    r2 = ax.bar(x + width/2, v_anh, width, label="Anhgoc (Full)", color="#ff7f0e", alpha=0.88, edgecolor="black")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.0, fontweight="bold", rotation=12)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", (r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    for r in r2:
        h = r.get_height()
        ax.annotate(f"{h:.3f}", (r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color="#b34700", fontweight="bold")

    if idx == 0:
        ax.legend(frameon=True, loc="upper left")

fig.suptitle("Can Gio Disturbance Benchmark: Cut vs. Anhgoc Across All Methods Including SOPSeg + LoRA", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
bench_plot_path = os.path.join(PLOTS_DIR, "sopseg_cut_vs_anhgoc_benchmark.png")
plt.savefig(bench_plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Done] Clean benchmark chart generated: {bench_plot_path}")
