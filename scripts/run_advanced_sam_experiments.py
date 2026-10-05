import os
import sys
import json
import argparse
from typing import Tuple, List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
import segmentation_models_pytorch as smp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.dataset import discover_scenes, CanGioDataset, BiTemporalCanGioDataset
from src.evaluate import compute_binary_metrics, stretch_rgb
from src.sam_satellite import (
    ensure_sam_checkpoint,
    apply_lora_to_sam,
    prepare_satellite_for_sam,
    generate_prompts,
    SAM_MODEL_TYPE
)
from src.temporal_models import build_bitemporal_model
from src.models import build_segmentation_model, AutoPadModelWrapper


def evaluate_sam_detailed(
    sam_model: nn.Module,
    scenes_list: list,
    device: torch.device,
    prompt_type: str = "multi_point",
    threshold: float = 0.5
) -> dict:
    """
    Evaluates SAM + LoRA with breakdown by year (2011 vs 2014) and combined.
    """
    sam_model.eval()
    ds = CanGioDataset(scenes_list, include_indices=False, augment=False)

    results_by_year = {"2011": {"preds": [], "targets": []}, "2014": {"preds": [], "targets": []}}
    all_preds, all_targets = [], []
    per_scene_metrics = []

    with torch.no_grad():
        for idx in range(len(ds)):
            img_tensor, mask_tensor, meta = ds[idx]
            gt_np = mask_tensor.numpy().squeeze()
            orig_size = (img_tensor.shape[1], img_tensor.shape[2])

            sam_img, _ = prepare_satellite_for_sam(img_tensor)
            sam_img = sam_img.unsqueeze(0).to(device)

            pts_np, lbls_np = generate_prompts(gt_np, orig_size, target_size=1024, prompt_type=prompt_type)
            pts_tensor = torch.from_numpy(pts_np).unsqueeze(0).to(device)
            lbls_tensor = torch.from_numpy(lbls_np).unsqueeze(0).to(device)

            image_embeddings = sam_model.image_encoder(sam_img)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=(pts_tensor, lbls_tensor),
                boxes=None,
                masks=None
            )
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            pred_mask = F.interpolate(low_res_masks, size=orig_size, mode="bilinear", align_corners=False)
            prob = torch.sigmoid(pred_mask).cpu().numpy().squeeze()

            # Record
            p_flat = prob.flatten()
            t_flat = gt_np.flatten()
            all_preds.append(p_flat)
            all_targets.append(t_flat)

            year_key = "2011" if "2011" in meta["year_folder"] else "2014"
            if year_key in results_by_year:
                results_by_year[year_key]["preds"].append(p_flat)
                results_by_year[year_key]["targets"].append(t_flat)

            sc_metrics = compute_binary_metrics(p_flat, t_flat, threshold=threshold)
            sc_metrics["scene_id"] = meta["scene_id"]
            sc_metrics["date"] = meta["date"]
            sc_metrics["sensor"] = meta["sensor"]
            per_scene_metrics.append(sc_metrics)

    # Combined metrics
    comb_preds = np.concatenate(all_preds)
    comb_targets = np.concatenate(all_targets)
    overall_metrics = compute_binary_metrics(comb_preds, comb_targets, threshold=threshold)

    # Breakdown metrics
    breakdown = {}
    for yr, data in results_by_year.items():
        if data["preds"]:
            y_p = np.concatenate(data["preds"])
            y_t = np.concatenate(data["targets"])
            breakdown[yr] = compute_binary_metrics(y_p, y_t, threshold=threshold)

    return {
        "overall": overall_metrics,
        "breakdown_by_year": breakdown,
        "per_scene": per_scene_metrics
    }


def train_sam_with_rank(
    rank: int = 8,
    num_epochs: int = 8,
    lr: float = 1e-4,
    data_source: str = "anhgoc",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[nn.Module, str]:
    """
    Trains SAM + LoRA with specific rank r on training years (2006, 2007, 2008, 2010).
    """
    print(f"\n[*] Training SAM + LoRA (Rank r={rank}) on Pre-2011 Scenes | Source: {data_source.upper()}...")
    ckpt_path = ensure_sam_checkpoint(MODELS_DIR)
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=ckpt_path).to(device)

    # Apply LoRA with rank r
    apply_lora_to_sam(sam, rank=rank, alpha=float(rank * 2))

    all_scenes = discover_scenes(os.path.join(PROJECT_ROOT, "data"), source=data_source)
    gt_scenes = [s for s in all_scenes if s.get("groundtruth_path") and os.path.exists(s["groundtruth_path"])]

    # Strictly split: Train on pre-2011, Validate on 2011 & 2014
    train_scenes = [s for s in gt_scenes if not any(vy in s["year_folder"] for vy in ["2011", "2014"])]
    val_scenes = [s for s in gt_scenes if any(vy in s["year_folder"] for vy in ["2011", "2014"])]

    print(f"Train scenes: {len(train_scenes)} | Held-Out Val scenes (2011+2014): {len(val_scenes)}")

    train_ds = CanGioDataset(train_scenes, include_indices=False, augment=True)

    optimizer = torch.optim.AdamW([p for p in sam.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    best_weights_path = os.path.join(MODELS_DIR, f"best_sam_lora_r{rank}.pth")

    for epoch in range(1, num_epochs + 1):
        sam.train()
        epoch_losses = []
        for idx in range(len(train_ds)):
            img_tensor, mask_tensor, _ = train_ds[idx]
            gt_np = mask_tensor.numpy().squeeze()
            orig_size = (img_tensor.shape[1], img_tensor.shape[2])

            sam_img, _ = prepare_satellite_for_sam(img_tensor)
            sam_img = sam_img.unsqueeze(0).to(device)

            pts_np, lbls_np = generate_prompts(gt_np, orig_size, target_size=1024, prompt_type="multi_point")
            pts_tensor = torch.from_numpy(pts_np).unsqueeze(0).to(device)
            lbls_tensor = torch.from_numpy(lbls_np).unsqueeze(0).to(device)

            optimizer.zero_grad()
            image_embeddings = sam.image_encoder(sam_img)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=(pts_tensor, lbls_tensor),
                boxes=None,
                masks=None
            )
            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            pred_mask = F.interpolate(low_res_masks, size=orig_size, mode="bilinear", align_corners=False)
            gt_target = mask_tensor.unsqueeze(0).to(device)

            loss = 0.5 * dice_loss_fn(pred_mask, gt_target) + 0.5 * bce_loss_fn(pred_mask, gt_target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))

        # Check validation
        val_eval = evaluate_sam_detailed(sam, val_scenes, device, prompt_type="multi_point")
        val_dice = val_eval["overall"]["dice"]
        val_iou = val_eval["overall"]["iou"]

        if epoch % 2 == 0 or epoch == num_epochs:
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss: {mean_loss:.4f} | Held-Out Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            trainable_state = {k: v for k, v in sam.state_dict().items() if "lora_" in k or "mask_decoder" in k}
            torch.save(trainable_state, best_weights_path)

    # Load best checkpoint
    state = torch.load(best_weights_path, map_location=device, weights_only=True)
    sam.load_state_dict(state, strict=False)
    print(f"[*] Training finished for SAM (r={rank}). Best Held-Out Val Dice: {best_val_dice:.4f}")
    return sam, best_weights_path


def evaluate_baselines_on_heldout(val_scenes: list, device: torch.device) -> dict:
    """
    Evaluates Siamese U-Net (ResNet-34) and Standard U-Net on the same 2011 + 2014 held-out scenes.
    """
    baselines_results = {}

    # 1. Siamese U-Net (ResNet-34)
    siamese_ckpt = os.path.join(MODELS_DIR, "best_option2_siamese_resnet34_cut.pth")
    if os.path.exists(siamese_ckpt):
        siamese_model = build_bitemporal_model("siamese", "resnet34", in_channels=9).to(device)
        siamese_model.load_state_dict(torch.load(siamese_ckpt, map_location=device, weights_only=True))
        siamese_model.eval()

        val_bi_ds = BiTemporalCanGioDataset(val_scenes, pairing_strategy="consecutive", augment=False)
        all_p, all_t = [], []
        with torch.no_grad():
            for idx in range(len(val_bi_ds)):
                t1, t2, mask, _ = val_bi_ds[idx]
                t1_in, t2_in = t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device)
                prob = torch.sigmoid(siamese_model(t1_in, t2_in)).cpu().numpy()[0, 0]
                all_p.append(prob.flatten())
                all_t.append(mask.numpy()[0].flatten())

        if all_p:
            p_cat = np.concatenate(all_p)
            t_cat = np.concatenate(all_t)
            baselines_results["Siamese U-Net (ResNet-34)"] = compute_binary_metrics(p_cat, t_cat)

    # 2. Standard Fully Supervised U-Net (ResNet-34)
    unet_ckpt = os.path.join(MODELS_DIR, "best_unet_resnet34.pth")
    if os.path.exists(unet_ckpt):
        unet_model = AutoPadModelWrapper(
            build_segmentation_model(arch="unet", encoder_name="resnet34", in_channels=9, classes=1),
            multiple=32
        ).to(device)
        try:
            unet_model.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=True))
        except Exception:
            pass  # Fallback if checkpoint structure differs
        unet_model.eval()

        std_ds = CanGioDataset(val_scenes, include_indices=True, augment=False)
        all_p, all_t = [], []
        with torch.no_grad():
            for idx in range(len(std_ds)):
                img, mask, _ = std_ds[idx]
                img_in = img.unsqueeze(0).to(device)
                prob = torch.sigmoid(unet_model(img_in)).cpu().numpy()[0, 0]
                all_p.append(prob.flatten())
                all_t.append(mask.numpy()[0].flatten())

        if all_p:
            p_cat = np.concatenate(all_p)
            t_cat = np.concatenate(all_t)
            baselines_results["Standard 2D U-Net (ResNet-34)"] = compute_binary_metrics(p_cat, t_cat)

    return baselines_results


def plot_sam_comparison(
    val_scene: dict,
    sam_r8: nn.Module,
    sam_r16: nn.Module,
    device: torch.device,
    save_path: str
):
    """
    Generates comparison figure: True Color, False Color, GT, SAM r=8, SAM r=16, Error Map.
    """
    ds = CanGioDataset([val_scene], include_indices=False, augment=False)
    img_tensor, mask_tensor, meta = ds[0]
    gt_np = mask_tensor.numpy().squeeze()
    orig_size = (img_tensor.shape[1], img_tensor.shape[2])

    sam_img, _ = prepare_satellite_for_sam(img_tensor)
    sam_img = sam_img.unsqueeze(0).to(device)

    pts_np, lbls_np = generate_prompts(gt_np, orig_size, target_size=1024, prompt_type="multi_point")
    pts_tensor = torch.from_numpy(pts_np).unsqueeze(0).to(device)
    lbls_tensor = torch.from_numpy(lbls_np).unsqueeze(0).to(device)

    def predict(model):
        model.eval()
        with torch.no_grad():
            image_embeddings = model.image_encoder(sam_img)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(pts_tensor, lbls_tensor),
                boxes=None,
                masks=None
            )
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            pred_mask = F.interpolate(low_res_masks, size=orig_size, mode="bilinear", align_corners=False)
            return torch.sigmoid(pred_mask).cpu().numpy().squeeze()

    prob_r8 = predict(sam_r8)
    prob_r16 = predict(sam_r16)

    img_np = img_tensor.numpy()
    rgb = stretch_rgb(np.stack([img_np[2], img_np[1], img_np[0]], axis=-1))
    nir = img_np[3] if img_np.shape[0] > 3 else img_np[2]
    false_color = stretch_rgb(np.stack([nir, img_np[2], img_np[1]], axis=-1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"SAM + LoRA Evaluation on Held-Out Scene: {meta['scene_id']} ({meta['date']}, {meta['sensor']})",
        fontsize=14, fontweight="bold"
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("True Color (RGB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(false_color)
    axes[0, 1].set_title("False Color (NIR-R-G)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gt_np, cmap="Blues", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Ground Truth (Positives: {int(gt_np.sum())})")
    axes[0, 2].axis("off")

    im_r8 = axes[1, 0].imshow(prob_r8, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("SAM + LoRA (r=8) Prediction")
    axes[1, 0].axis("off")
    fig.colorbar(im_r8, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_r16 = axes[1, 1].imshow(prob_r16, cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("SAM + LoRA (r=16) Prediction")
    axes[1, 1].axis("off")
    fig.colorbar(im_r16, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Confusion overlay for r=16
    bin_pred = (prob_r16 >= 0.5).astype(np.float32)
    h, w = gt_np.shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    tp = (bin_pred == 1) & (gt_np == 1)
    fp = (bin_pred == 1) & (gt_np == 0)
    fn = (bin_pred == 0) & (gt_np == 1)
    overlay[tp] = [0.0, 0.9, 0.0]  # Green: TP
    overlay[fp] = [0.9, 0.1, 0.1]  # Red: FP
    overlay[fn] = [0.1, 0.4, 0.9]  # Blue: FN

    axes[1, 2].imshow(false_color * 0.4 + overlay * 0.6)
    axes[1, 2].set_title("Overlay r=16 (Green: TP, Red: FP, Blue: FN)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SAM + LoRA Evaluation on Held-Out Validation Years (2011 & 2014)")
    parser.add_argument("--source", choices=["cut", "anhgoc"], default="anhgoc", help="Data source: cut or anhgoc")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset Source: {args.source.upper()}")

    all_scenes = discover_scenes(os.path.join(PROJECT_ROOT, "data"), source=args.source)
    gt_scenes = [s for s in all_scenes if s.get("groundtruth_path") and os.path.exists(s["groundtruth_path"])]

    # Strictly held-out validation years: 2011 and 2014
    val_scenes = [s for s in gt_scenes if any(vy in s["year_folder"] for vy in ["2011", "2014"])]
    print(f"Held-Out Validation Scenes ({len(val_scenes)}): {[s['scene_id'] for s in val_scenes]}")

    # 1. Train and Evaluate SAM + LoRA (r=8)
    sam_r8, ckpt_r8 = train_sam_with_rank(rank=8, num_epochs=args.epochs, data_source=args.source, device=device)
    eval_r8_multi = evaluate_sam_detailed(sam_r8, val_scenes, device, prompt_type="multi_point")
    eval_r8_single = evaluate_sam_detailed(sam_r8, val_scenes, device, prompt_type="single_point")

    # 2. Train and Evaluate SAM + LoRA (r=16)
    sam_r16, ckpt_r16 = train_sam_with_rank(rank=16, num_epochs=args.epochs, data_source=args.source, device=device)
    eval_r16_multi = evaluate_sam_detailed(sam_r16, val_scenes, device, prompt_type="multi_point")
    eval_r16_single = evaluate_sam_detailed(sam_r16, val_scenes, device, prompt_type="single_point")

    # 3. Evaluate Baselines on the exact same held-out validation set
    baseline_results = evaluate_baselines_on_heldout(val_scenes, device)

    # 4. Generate visual comparison plots on held-out 2011 and 2014 scenes
    sc_2011 = [s for s in val_scenes if "2011" in s["year_folder"]][0]
    sc_2014 = [s for s in val_scenes if "2014" in s["year_folder"]][0]
    plot_sam_comparison(sc_2011, sam_r8, sam_r16, device, os.path.join(PLOTS_DIR, f"eval_heldout_2011_landsat5_sam_{args.source}.png"))
    plot_sam_comparison(sc_2014, sam_r8, sam_r16, device, os.path.join(PLOTS_DIR, f"eval_heldout_2014_landsat8_sam_{args.source}.png"))

    # Compile structured summary
    summary_data = {
        "data_source": args.source,
        "held_out_validation_years": ["2011 (Landsat-5 TM)", "2014 (Landsat-8 OLI)"],
        "num_val_scenes": len(val_scenes),
        "models": {
            "SAM + LoRA (r=8, multi-point)": eval_r8_multi["overall"],
            "SAM + LoRA (r=8, single-point)": eval_r8_single["overall"],
            "SAM + LoRA (r=16, multi-point)": eval_r16_multi["overall"],
            "SAM + LoRA (r=16, single-point)": eval_r16_single["overall"],
        },
        "breakdown_by_year": {
            "SAM_r8_multi": eval_r8_multi["breakdown_by_year"],
            "SAM_r16_multi": eval_r16_multi["breakdown_by_year"]
        },
        "baselines_on_heldout": baseline_results
    }

    out_json = os.path.join(RESULTS_DIR, f"advanced_sam_validation_results_{args.source}.json")
    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[*] HELD-OUT VALIDATION BENCHMARK (YEARS 2011 & 2014) | SOURCE: {args.source.upper()}")
    print("=" * 80)
    print(f"{'Model Architecture':<34} | {'Val Dice/F1':<12} | {'Val IoU':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 80)

    for name, m in summary_data["models"].items():
        print(f"{name:<34} | {m['dice']:<12.4f} | {m['iou']:<10.4f} | {m['precision']*100:<9.2f}% | {m['recall']*100:<9.2f}%")

    for name, m in baseline_results.items():
        print(f"{name:<34} | {m['dice']:<12.4f} | {m['iou']:<10.4f} | {m['precision']*100:<9.2f}% | {m['recall']*100:<9.2f}%")

    print("=" * 80)
    print(f"Results saved to: {out_json}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
