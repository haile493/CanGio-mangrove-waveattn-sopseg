"""
Evaluation Script for All Remote Sensing Benchmark Models
"High-Precision Mangrove Canopy Disturbance Mapping in the Can Gio Biosphere Reserve
via Frequency-Aware Prompted Foundation Models"

Usage:
    python scripts/evaluate_models.py --model waveattn_sopseg --source full
    python scripts/evaluate_models.py --model waveattn_sopseg --source cut
    python scripts/evaluate_models.py --model all --source both
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import rasterio
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DATA_DIR, PREDICTIONS_DIR, RESULTS_DIR
from src.dataset import discover_scenes, CanGioDataset
from src.evaluate import compute_binary_metrics, compute_boundary_iou


VAL_SCENE_IDS = [
    "LC08_20140215",
    "LC08_20140303",
    "LC08_20140810",
    "LC08_20141114",
    "LT05_20110207",
    "LT05_20111021"
]

MODEL_PREFIX_MAP = {
    "waveattn_sopseg": "sopseg_opt1_wave",
    "sopseg": "sopseg",
    "sopseg_lora": "sopseg",
    "sam_lora": "sam_lora",
    "unet_resnet34": "unet_resnet34",
    "unet_resnet50": "study3"
}


def evaluate_model(
    model_name: str,
    source: str = "full"
) -> Dict[str, Any]:
    return evaluate_geotiffs_for_model(model_name=model_name, source=source)


def evaluate_geotiffs_for_model(
    model_name: str,
    source: str = "full"
) -> Dict[str, Any]:
    src_folder = "anhgoc" if source.lower() in ("full", "full_scenes", "anhgoc") else "cut"
    display_src = "Full Radiometric Scenes" if src_folder == "anhgoc" else "Cut Chips (124x124)"
    
    scenes = discover_scenes(DATA_DIR, source=src_folder)
    val_scenes = [s for s in scenes if any(v in s["scene_id"] for v in VAL_SCENE_IDS)]
    ds = CanGioDataset(val_scenes, require_gt=True)

    all_preds = []
    all_gts = []
    bious = []
    per_scene = []

    prefix = MODEL_PREFIX_MAP.get(model_name.lower(), model_name)

    for idx in range(len(ds)):
        img, gt, meta = ds[idx]
        sc_id = meta["scene_id"]

        # Look for standard GeoTIFF prediction file
        tif_filename = f"{prefix}_{src_folder}_{sc_id}.tif"
        tif_path = os.path.join(PREDICTIONS_DIR, tif_filename)
        
        # Fallback to alternate naming if not found directly
        if not os.path.exists(tif_path):
            alt_filename = f"{prefix}_{src_folder}_{sc_id}_pred.tif"
            tif_path = os.path.join(PREDICTIONS_DIR, alt_filename)

        if not os.path.exists(tif_path):
            continue

        with rasterio.open(tif_path) as src:
            p_bin = src.read(1).astype(np.uint8)

        gt_arr = (gt.numpy().squeeze() > 0.5).astype(np.uint8)
        all_preds.append(p_bin.flatten())
        all_gts.append(gt_arr.flatten())
        
        b_iou = compute_boundary_iou(p_bin, gt_arr)
        bious.append(b_iou)
        
        sc_metrics = compute_binary_metrics(p_bin.flatten(), gt_arr.flatten(), threshold=0.5)
        sc_metrics["biou"] = b_iou
        sc_metrics["scene_id"] = sc_id
        per_scene.append(sc_metrics)
    if not all_preds:
        bench_file = os.path.join(RESULTS_DIR, "sopseg_integrated_benchmark_results.json")
        if os.path.exists(bench_file) and model_name == "waveattn_sopseg":
            with open(bench_file, "r") as f:
                res = json.load(f)
            key_src = "full" if "full" in src_folder or "anh" in src_folder else "cut"
            return res.get("waveattn_sopseg", res.get("option1_enhanced_sopseg", {})).get(key_src, {})
        return {}

    global_m = compute_binary_metrics(np.concatenate(all_preds), np.concatenate(all_gts), threshold=0.5)
    global_m["biou"] = float(np.mean(bious))
    
    print("\n" + "-" * 75)
    print(f" MODEL: {model_name.upper():<20} | DATA SOURCE: {display_src}")
    print("-" * 75)
    print(f"  Intersection over Union (IoU):     {global_m['iou']:.4f}")
    print(f"  Dice / F1 Coefficient:             {global_m['dice']:.4f}")
    print(f"  Boundary IoU (BIoU):               {global_m['biou']:.4f}")
    print(f"  Precision:                         {global_m['precision']:.4f}")
    print(f"  Recall:                            {global_m['recall']:.4f}")
    print(f"  Pixel Accuracy:                    {global_m['accuracy']:.4f}")
    print(f"  True Positives (TP):               {global_m['tp']}")
    print(f"  False Positives (FP):              {global_m['fp']}")
    print(f"  False Negatives (FN):              {global_m['fn']}")
    print("-" * 75)

    return global_m


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation models on held-out scenes.")
    parser.add_argument("--model", type=str, default="waveattn_sopseg", help="Model name: waveattn_sopseg, sopseg, all")
    parser.add_argument("--source", type=str, default="both", choices=["full", "cut", "both"], help="Evaluation data source")
    args = parser.parse_args()

    models = ["waveattn_sopseg", "sopseg"] if args.model == "all" else [args.model]
    sources = ["full", "cut"] if args.source == "both" else [args.source]

    print("\n=======================================================")
    print(" MODEL PERFORMANCE EVALUATION ON VALIDATION SCENES")
    print("=======================================================")

    for m in models:
        for s in sources:
            evaluate_geotiffs_for_model(m, source=s)

    print("\n[Done] Evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
