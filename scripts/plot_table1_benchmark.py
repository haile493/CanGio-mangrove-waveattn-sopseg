import os
import numpy as np
import matplotlib.pyplot as plt

def generate_figure3_table1_comparison(output_path: str):
    # Set publication styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # -------------------------------------------------------------------------
    # DATA FROM TABLE 1 IN MANUSCRIPT
    # -------------------------------------------------------------------------
    # Metrics order
    metrics = ["IoU", "Boundary IoU", "Dice / F1", "Precision", "Recall"]
    
    # Cut Chips Models & Values
    cut_models = [
        "U-Net (ResNet-34)",
        "U-Net (ResNet-50)",
        "SAM + LoRA",
        "SOPSeg (Frozen)",
        "SOPSeg + LoRA",
        "WaveAttn SOPSeg (Ours)"
    ]
    cut_data = {
        "U-Net (ResNet-34)":       [0.3850, 0.2810, 0.5559, 0.4299, 0.7865],
        "U-Net (ResNet-50)":       [0.4444, 0.3120, 0.6154, 0.6243, 0.6067],
        "SAM + LoRA":              [0.5732, 0.4350, 0.7287, 0.8916, 0.6161],
        "SOPSeg (Frozen)":         [0.6980, 0.9619, 0.8222, 0.7363, 0.9307],
        "SOPSeg + LoRA":           [0.5872, 0.9148, 0.7399, 0.6506, 0.8577],
        "WaveAttn SOPSeg (Ours)":  [0.6468, 0.9419, 0.7855, 0.6785, 0.9326]
    }
    
    # Full Radiometric Scenes Models & Values
    full_models = [
        "U-Net (ResNet-34)",
        "U-Net (ResNet-50)",
        "SAM + LoRA",
        "SOPSeg (Frozen)",
        "SOPSeg + LoRA",
        "WaveAttn SOPSeg (Ours)"
    ]
    full_data = {
        "U-Net (ResNet-34)":       [0.3782, 0.2740, 0.5488, 0.4236, 0.7790],
        "U-Net (ResNet-50)":       [0.0801, 0.0984, 0.1484, 0.0867, 0.5131],
        "SAM + LoRA":              [0.5413, 0.4180, 0.7024, 0.8980, 0.5768],
        "SOPSeg (Frozen)":         [0.7378, 0.9641, 0.8492, 0.7755, 0.9382],
        "SOPSeg + LoRA":           [0.7474, 0.9654, 0.8554, 0.7914, 0.9307],
        "WaveAttn SOPSeg (Ours)":  [0.7576, 0.9634, 0.8621, 0.7987, 0.9363]
    }

    # Distinct, professional color palette
    model_colors = {
        "U-Net (ResNet-34)":      "#64748B",  # Slate
        "U-Net (ResNet-50)":      "#0284C7",  # Sky Blue
        "SAM + LoRA":             "#F59E0B",  # Amber
        "SOPSeg (Frozen)":        "#8B5CF6",  # Violet
        "SOPSeg + LoRA":          "#0D9488",  # Teal
        "WaveAttn SOPSeg (Ours)": "#15803D"   # Deep Forest Green (Highlight)
    }

    # Create 2-row layout: Top = Cut Chips, Bottom = Anhgoc Full
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), dpi=300)
    
    # -------------------------------------------------------------------------
    # PANEL (a): CUT CHIPS
    # -------------------------------------------------------------------------
    x = np.arange(len(metrics))
    n_cut = len(cut_models)
    bar_width_cut = 0.135
    offsets_cut = np.linspace(-(n_cut - 1) / 2 * bar_width_cut, (n_cut - 1) / 2 * bar_width_cut, n_cut)

    for i, m_name in enumerate(cut_models):
        vals = cut_data[m_name]
        is_ours = "Ours" in m_name
        edge_c = "#052e16" if is_ours else "black"
        lw = 1.3 if is_ours else 0.7
        alpha = 0.95 if is_ours else 0.85
        
        bars = ax1.bar(
            x + offsets_cut[i], vals, bar_width_cut,
            label=m_name,
            color=model_colors[m_name],
            edgecolor=edge_c,
            linewidth=lw,
            alpha=alpha
        )
        
        # Add numerical labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2.5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.0,
                rotation=90,
                fontweight="bold" if is_ours else "normal",
                color="#047857" if is_ours else "#1e293b"
            )

    ax1.set_title("(a) Model Performance on Cut Chips (124 × 124 pixels)", fontsize=13, fontweight="bold", pad=12, color="#1e3a8a")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1", fontsize=9.5, ncol=3)

    # -------------------------------------------------------------------------
    # PANEL (b): FULL RADIOMETRIC SCENES
    # -------------------------------------------------------------------------
    n_full = len(full_models)
    bar_width_full = 0.135
    offsets_full = np.linspace(-(n_full - 1) / 2 * bar_width_full, (n_full - 1) / 2 * bar_width_full, n_full)

    for i, m_name in enumerate(full_models):
        vals = full_data[m_name]
        is_ours = "Ours" in m_name
        edge_c = "#052e16" if is_ours else "black"
        lw = 1.3 if is_ours else 0.7
        alpha = 0.95 if is_ours else 0.85
        
        bars = ax2.bar(
            x + offsets_full[i], vals, bar_width_full,
            label=m_name,
            color=model_colors[m_name],
            edgecolor=edge_c,
            linewidth=lw,
            alpha=alpha
        )
        
        # Add numerical labels
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2.5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.0,
                rotation=90,
                fontweight="bold" if is_ours else "normal",
                color="#047857" if is_ours else "#1e293b"
            )

    ax2.set_title("(b) Model Performance on Full Radiometric Scenes", fontsize=13, fontweight="bold", pad=12, color="#1e3a8a")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1", fontsize=9.5, ncol=3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure 3 successfully generated at: {output_path}")

if __name__ == "__main__":
    out_p = "results/plots/table1_models_benchmark_cut_vs_full.png"
    generate_figure3_table1_comparison(out_p)
