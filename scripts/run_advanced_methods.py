import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.models import get_model_from_config
from src.evaluate import evaluate_model, compute_binary_metrics, stretch_rgb
from src.superpixel_wsl import train_superpixel_wsl
from src.sam_satellite import (
    train_sam_lora,
    evaluate_sam,
    prepare_satellite_for_sam,
    generate_prompts,
    sam_model_registry,
    apply_lora_to_sam,
    SAM_MODEL_TYPE,
    ensure_sam_checkpoint
)


def run_comparative_benchmark(config: Config, force_retrain: bool = False):
    """
    Executes a head-to-head comparison between:
    1. Baseline: Fully Supervised U-Net
    2. Study 1: Superpixel-Assisted Weakly Supervised Learning (Rivier et al.)
    3. Study 2: Segment Anything Model with LoRA Adaptation (Scribano et al.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 75)
    print("CAN GIO SATELLITE SEGMENTATION: ADVANCED METHODS BENCHMARK")
    print(f"Device: {device} | Source: {config.data_source} | Holdout Validation Years: {config.val_years}")
    print("=" * 75)

    # Common validation dataset
    all_scenes = discover_scenes(config.data_dir, source=config.data_source)
    val_scenes = [s for s in all_scenes if any(vy in s["year_folder"] for vy in config.val_years) and s.get("groundtruth_path")]
    val_dataset_full = CanGioDataset(val_scenes, spectral_bands=config.spectral_bands, include_indices=True, augment=False, study_bounds=config.study_bounds)
    val_dataset_sam = CanGioDataset(val_scenes, spectral_bands=config.spectral_bands, include_indices=False, augment=False, study_bounds=config.study_bounds)
    val_loader = DataLoader(val_dataset_full, batch_size=config.batch_size, shuffle=False)

    results_table = []

    # -------------------------------------------------------------
    # 1. Baseline: Fully Supervised U-Net
    # -------------------------------------------------------------
    print("\n>>> [1/3] Evaluating Fully Supervised U-Net Baseline...")
    baseline_model = get_model_from_config(config).to(device)
    baseline_ckpt = os.path.join(config.models_dir, f"best_{config.arch}_{config.encoder_name}.pth")
    if os.path.exists(baseline_ckpt) and not force_retrain:
        ckpt = torch.load(baseline_ckpt, map_location=device, weights_only=False)
        baseline_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded existing baseline checkpoint from {baseline_ckpt}")
    else:
        print("Training baseline on Anhgoc data for 20 epochs...")
        from src.train import train_model
        baseline_model, _ = train_model(config, num_epochs=20)

    base_metrics, _ = evaluate_model(baseline_model, val_loader, device=device, threshold=config.threshold)
    results_table.append({
        "Method": "Fully-Supervised U-Net",
        "Supervision": "Full Mask (100% pixels)",
        "Annotation Cost": "Dense Polygons (High)",
        "IoU": base_metrics["iou"],
        "Dice/F1": base_metrics["dice"],
        "Precision": base_metrics["precision"],
        "Recall": base_metrics["recall"],
        "Accuracy": base_metrics["accuracy"]
    })

    # -------------------------------------------------------------
    # 2. Study 1: Superpixel-Assisted WSL (Rivier et al.)
    # -------------------------------------------------------------
    print("\n>>> [2/3] Training & Evaluating Superpixel WSL (Study 1)...")
    wsl_ckpt = os.path.join(config.models_dir, "best_superpixel_wsl.pth")
    wsl_model = get_model_from_config(config).to(device)
    if os.path.exists(wsl_ckpt) and not force_retrain:
        wsl_model.load_state_dict(torch.load(wsl_ckpt, map_location=device, weights_only=True))
        print(f"Loaded existing Superpixel WSL model from {wsl_ckpt}")
    else:
        print("Training Superpixel WSL on Anhgoc data...")
        wsl_model, _ = train_superpixel_wsl(config, n_points=50, n_segments=100, num_epochs=12)

    wsl_metrics, _ = evaluate_model(wsl_model, val_loader, device=device, threshold=config.threshold)
    results_table.append({
        "Method": "Superpixel WSL (Study 1)",
        "Supervision": "50 Query Points + SLIC Superpixels",
        "Annotation Cost": "50 Clicks / Scene (~95% cheaper)",
        "IoU": wsl_metrics["iou"],
        "Dice/F1": wsl_metrics["dice"],
        "Precision": wsl_metrics["precision"],
        "Recall": wsl_metrics["recall"],
        "Accuracy": wsl_metrics["accuracy"]
    })

    # -------------------------------------------------------------
    # 3. Study 2: SAM + LoRA (Scribano et al.)
    # -------------------------------------------------------------
    print("\n>>> [3/3] Training & Evaluating SAM + LoRA (Study 2)...")
    sam_ckpt = os.path.join(config.models_dir, "best_sam_lora.pth")
    sam_base_path = ensure_sam_checkpoint(config.models_dir)
    sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=sam_base_path).to(device)
    apply_lora_to_sam(sam_model, rank=8, alpha=16.0)

    if os.path.exists(sam_ckpt) and not force_retrain:
        state = torch.load(sam_ckpt, map_location=device, weights_only=True)
        sam_model.load_state_dict(state, strict=False)
        print(f"Loaded existing SAM LoRA weights from {sam_ckpt}")
    else:
        print("Fine-tuning SAM with LoRA on Anhgoc data...")
        sam_model, _ = train_sam_lora(config, num_epochs=6, lora_rank=8, prompt_type="multi_point")

    sam_metrics = evaluate_sam(sam_model, val_dataset_sam, device=device, prompt_type="multi_point", threshold=config.threshold)
    results_table.append({
        "Method": "SAM + LoRA (Study 2)",
        "Supervision": "Prompt-based (Centroid + Borders)",
        "Annotation Cost": "Interactive Foundation Model",
        "IoU": sam_metrics["iou"],
        "Dice/F1": sam_metrics["dice"],
        "Precision": sam_metrics["precision"],
        "Recall": sam_metrics["recall"],
        "Accuracy": sam_metrics["accuracy"]
    })

    # -------------------------------------------------------------
    # Print Benchmark Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Method':<28} | {'Annotation Cost':<22} | {'IoU':<7} | {'Dice (F1)':<9} | {'Recall':<7} | {'Precision':<9}")
    print("-" * 90)
    for row in results_table:
        print(f"{row['Method']:<28} | {row['Annotation Cost']:<22} | {row['IoU']:.4f}  | {row['Dice/F1']:.4f}    | {row['Recall']:.4f}  | {row['Precision']:.4f}")
    print("=" * 90)

    # Save benchmark table as JSON
    benchmark_json_path = os.path.join(config.results_dir, "benchmark_comparison.json")
    with open(benchmark_json_path, "w") as f:
        json.dump(results_table, f, indent=2)

    # -------------------------------------------------------------
    # Visual Comparison Plot across all methods
    # -------------------------------------------------------------
    print("\nGenerating multi-method visual comparison figures...")
    sample_img_full, sample_gt_full, sample_meta = val_dataset_full[0]
    sample_img_sam, _, _ = val_dataset_sam[0]

    # Baseline Prediction
    with torch.no_grad():
        b_logits = baseline_model(sample_img_full.unsqueeze(0).to(device))
        b_prob = torch.sigmoid(b_logits).cpu().numpy().squeeze()

    # Superpixel WSL Prediction
    with torch.no_grad():
        w_logits = wsl_model(sample_img_full.unsqueeze(0).to(device))
        w_prob = torch.sigmoid(w_logits).cpu().numpy().squeeze()

    # SAM LoRA Prediction
    with torch.no_grad():
        sam_in, orig_sz = prepare_satellite_for_sam(sample_img_sam)
        pts_np, lbls_np = generate_prompts(sample_gt_full.numpy().squeeze(), orig_sz, prompt_type="multi_point")
        pts_t = torch.from_numpy(pts_np).unsqueeze(0).to(device)
        lbls_t = torch.from_numpy(lbls_np).unsqueeze(0).to(device)

        feat = sam_model.image_encoder(sam_in.unsqueeze(0).to(device))
        sp, dn = sam_model.prompt_encoder(points=(pts_t, lbls_t), boxes=None, masks=None)
        masks, _ = sam_model.mask_decoder(
            image_embeddings=feat,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dn,
            multimask_output=False
        )
        sam_pred = F.interpolate(masks, size=orig_sz, mode="bilinear", align_corners=False)
        sam_prob = torch.sigmoid(sam_pred).cpu().numpy().squeeze()

    # RGB Composite for display
    img_np = sample_img_full.numpy()
    rgb = np.stack([img_np[2], img_np[1], img_np[0]], axis=-1)
    rgb_disp = stretch_rgb(rgb)
    gt_disp = sample_gt_full.numpy().squeeze()

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(f"Can Gio Disturbance Segmentation Comparison - {sample_meta['scene_id']}", fontsize=14, fontweight="bold")

    axes[0].imshow(rgb_disp)
    axes[0].set_title("True Color (RGB)")
    axes[0].axis("off")

    axes[1].imshow(gt_disp, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"Ground Truth\n({int(gt_disp.sum())} pixels)")
    axes[1].axis("off")

    axes[2].imshow(b_prob, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title(f"Baseline: Full U-Net\n(Dice: {base_metrics['dice']:.4f})")
    axes[2].axis("off")

    axes[3].imshow(w_prob, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title(f"Study 1: Superpixel WSL\n(Dice: {wsl_metrics['dice']:.4f})")
    axes[3].axis("off")

    axes[4].imshow(sam_prob, cmap="magma", vmin=0, vmax=1)
    axes[4].set_title(f"Study 2: SAM + LoRA\n(Dice: {sam_metrics['dice']:.4f})")
    axes[4].axis("off")

    plt.tight_layout()
    comp_plot_path = os.path.join(config.plots_dir, "method_comparison.png")
    plt.savefig(comp_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {comp_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Can Gio Segmentation - Advanced Methods (Study 1 & Study 2)")
    parser.add_argument("--method", type=str, choices=["superpixel", "sam_lora", "compare"], default="compare")
    parser.add_argument("--source", type=str, default="anhgoc", choices=["anhgoc", "cut", "auto"], help="Imagery source")
    parser.add_argument("--compare", action="store_true", help="Run full comparison across baseline and both studies")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of models on the current data source")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    config = Config(data_source=args.source)

    if args.method == "superpixel" and not args.compare:
        train_superpixel_wsl(config, num_epochs=args.epochs)
    elif args.method == "sam_lora" and not args.compare:
        train_sam_lora(config, num_epochs=args.epochs)
    elif args.compare or args.method == "compare":
        run_comparative_benchmark(config, force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
