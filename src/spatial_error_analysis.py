import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def analyze_spatial_errors(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    boundary_radius: int = 2
) -> Dict[str, float]:
    """
    Implements the Error-Centric Spatial Analysis protocol from
    Dinh et al. (2026), Section 3.6 & 5.3:
    Categorizes spatial prediction errors into 4 distinct groups:
    1. Under-segmentation Errors: Two or more adjacent GT entities merged into one predicted polygon.
    2. False Negative Errors: Actual target completely missed (internal omission).
    3. Boundary Accuracy Errors: Deviations located along the contour transition zone (+- boundary_radius px).
    4. False Positive Errors: Spurious detections in non-target background areas.
    """
    pred_bool = (pred_bin > 0.5)
    gt_bool = (gt_bin > 0.5)

    fp_mask = pred_bool & (~gt_bool)
    fn_mask = (~pred_bool) & gt_bool
    total_error_pixels = int(fp_mask.sum() + fn_mask.sum())

    if total_error_pixels == 0:
        return {
            "under_segmentation": 0,
            "false_negative": 0,
            "boundary": 0,
            "false_positive": 0,
            "total_error": 0,
            "pct_under": 0.0,
            "pct_fn": 0.0,
            "pct_bound": 0.0,
            "pct_fp": 0.0
        }

    # 1. Boundary transition zone (+- boundary_radius px along GT contour)
    gt_dilated = ndi.binary_dilation(gt_bool, iterations=boundary_radius)
    gt_eroded = ndi.binary_erosion(gt_bool, iterations=boundary_radius)
    gt_boundary_band = gt_dilated & (~gt_eroded)

    # 2. Under-segmentation identification
    gt_labeled, num_gt = ndi.label(gt_bool)
    under_seg_mask = np.zeros_like(gt_bool, dtype=bool)

    if num_gt > 1:
        pred_labeled, num_preds = ndi.label(pred_bool)
        for p_id in range(1, num_preds + 1):
            pred_comp = (pred_labeled == p_id)
            gt_ids = np.unique(gt_labeled[pred_comp])
            gt_ids = gt_ids[gt_ids > 0]
            if len(gt_ids) > 1:
                # Prediction connects multiple separate GT entities:
                # The bridging false-positive pixels between them represent under-segmentation
                under_seg_mask |= (pred_comp & fp_mask)

    # 3. Categorize error pixels disjointly so they partition total_error_pixels
    is_under_seg = under_seg_mask
    remaining_fp = fp_mask & (~is_under_seg)
    remaining_fn = fn_mask

    is_boundary = (remaining_fp | remaining_fn) & gt_boundary_band
    is_pure_fn = remaining_fn & (~is_boundary)
    is_pure_fp = remaining_fp & (~is_boundary)

    n_under = int(is_under_seg.sum())
    n_fn = int(is_pure_fn.sum())
    n_bound = int(is_boundary.sum())
    n_fp = int(is_pure_fp.sum())
    total_classified = n_under + n_fn + n_bound + n_fp

    return {
        "under_segmentation": n_under,
        "false_negative": n_fn,
        "boundary": n_bound,
        "false_positive": n_fp,
        "total_error": total_classified,
        "pct_under": (n_under / total_classified) * 100.0 if total_classified > 0 else 0.0,
        "pct_fn": (n_fn / total_classified) * 100.0 if total_classified > 0 else 0.0,
        "pct_bound": (n_bound / total_classified) * 100.0 if total_classified > 0 else 0.0,
        "pct_fp": (n_fp / total_classified) * 100.0 if total_classified > 0 else 0.0,
        "masks": {
            "under_segmentation": is_under_seg,
            "false_negative": is_pure_fn,
            "boundary": is_boundary,
            "false_positive": is_pure_fp
        }
    }


def aggregate_spatial_errors(results_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregates spatial error metrics across multiple tiles/scenes."""
    total_under = sum(r["under_segmentation"] for r in results_list)
    total_fn = sum(r["false_negative"] for r in results_list)
    total_bound = sum(r["boundary"] for r in results_list)
    total_fp = sum(r["false_positive"] for r in results_list)
    total_err = total_under + total_fn + total_bound + total_fp

    if total_err == 0:
        return {
            "under_segmentation": 0,
            "false_negative": 0,
            "boundary": 0,
            "false_positive": 0,
            "total_error": 0,
            "pct_under": 0.0,
            "pct_fn": 0.0,
            "pct_bound": 0.0,
            "pct_fp": 0.0
        }

    return {
        "under_segmentation": total_under,
        "false_negative": total_fn,
        "boundary": total_bound,
        "false_positive": total_fp,
        "total_error": total_err,
        "pct_under": (total_under / total_err) * 100.0,
        "pct_fn": (total_fn / total_err) * 100.0,
        "pct_bound": (total_bound / total_err) * 100.0,
        "pct_fp": (total_fp / total_err) * 100.0
    }


def plot_spatial_error_shift(
    pretrained_errors: Dict[str, float],
    finetuned_errors: Dict[str, float],
    save_path: str,
    title_suffix: str = ""
):
    """
    Renders the stacked bar chart of urban geographic spatial error shift,
    faithfully replicating Figure 4 from Dinh et al. (2026).
    """
    categories = ["Under-segmentation", "False Negative", "Boundary Error", "False Positive"]
    colors = ["#e66101", "#5e3c99", "#e7298a", "#4daf4a"]  # Color scheme matching study

    labels = ["Pre-trained\n(Zero-shot)", "Fine-tuned\n(Adapted)"]
    p_vals = [
        pretrained_errors["pct_under"],
        pretrained_errors["pct_fn"],
        pretrained_errors["pct_bound"],
        pretrained_errors["pct_fp"]
    ]
    f_vals = [
        finetuned_errors["pct_under"],
        finetuned_errors["pct_fn"],
        finetuned_errors["pct_bound"],
        finetuned_errors["pct_fp"]
    ]

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    x = np.arange(len(labels))
    width = 0.45

    bottom_p = 0.0
    bottom_f = 0.0

    for idx, cat in enumerate(categories):
        val_p = p_vals[idx]
        val_f = f_vals[idx]

        rect_p = ax.bar(x[0], val_p, width, bottom=bottom_p, color=colors[idx], label=cat, edgecolor="white", linewidth=1.2)
        rect_f = ax.bar(x[1], val_f, width, bottom=bottom_f, color=colors[idx], edgecolor="white", linewidth=1.2)

        # Value annotations
        if val_p >= 3.0:
            ax.text(x[0], bottom_p + val_p / 2.0, f"{val_p:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        if val_f >= 3.0:
            ax.text(x[1], bottom_f + val_f / 2.0, f"{val_f:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

        bottom_p += val_p
        bottom_f += val_f

    ax.set_ylabel("Error Pixel Distribution (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Shift in Spatial Error Distribution Before and After Fine-Tuning\n{title_suffix}", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Spatial error shift plot saved to: {save_path}")
