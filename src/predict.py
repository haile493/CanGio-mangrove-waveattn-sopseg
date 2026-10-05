import os
import glob
from typing import Dict, Optional, List
import numpy as np
import rasterio
from rasterio.transform import Affine
import torch
import torch.nn as nn

from src.config import Config, PREDICTIONS_DIR
from src.dataset import CanGioDataset, discover_scenes
from src.models import get_model_from_config


def export_geotiff(
    data: np.ndarray,
    output_path: str,
    crs: str,
    transform: Optional[Affine] = None,
    nodata: Optional[float] = None
):
    """
    Exports a 2D numpy array to a georeferenced GeoTIFF file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = data.shape
    dtype = rasterio.float32 if data.dtype == np.float32 else rasterio.uint8

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "compress": "lzw"
    }
    if transform:
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)


def predict_scene(
    model: nn.Module,
    scene: Dict,
    config: Config,
    device: torch.device,
    save_dir: str = PREDICTIONS_DIR,
    threshold: float = 0.5
) -> Dict[str, str]:
    """
    Runs model inference on a single satellite scene and exports:
    - Probability GeoTIFF (0.0 to 1.0)
    - Binary mask GeoTIFF (0 or 1)
    """
    model.eval()
    dataset = CanGioDataset(
        scenes=[scene],
        spectral_bands=config.spectral_bands,
        include_indices=config.include_indices,
        indices=config.indices,
        augment=False,
        require_gt=False,
        study_bounds=config.study_bounds
    )
    
    image_tensor, _, meta = dataset[0]
    input_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_batch)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)

    binary_mask = (prob >= threshold).astype(np.uint8)

    scene_id = meta["scene_id"]
    crs = meta["crs"]
    transform = Affine(*meta["transform"][:6]) if meta["transform"] else None

    prob_path = os.path.join(save_dir, f"{scene_id}_disturbance_prob.tif")
    mask_path = os.path.join(save_dir, f"{scene_id}_disturbance_mask.tif")

    export_geotiff(prob.astype(np.float32), prob_path, crs=crs, transform=transform)
    export_geotiff(binary_mask, mask_path, crs=crs, transform=transform)

    return {
        "scene_id": scene_id,
        "prob_geotiff": prob_path,
        "mask_geotiff": mask_path,
        "positive_pixels": int(binary_mask.sum()),
        "positive_percentage": float(binary_mask.mean() * 100.0)
    }


def predict_all_scenes(
    model: nn.Module,
    config: Config,
    device: torch.device,
    save_dir: str = PREDICTIONS_DIR,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Discovers all scenes in data directory (2006 to 2014) and runs inference on all.
    """
    scenes = discover_scenes(config.data_dir, source=config.data_source)
    print(f"Running inference on {len(scenes)} total satellite scenes...")
    results = []
    for sc in scenes:
        res = predict_scene(
            model=model,
            scene=sc,
            config=config,
            device=device,
            save_dir=save_dir,
            threshold=threshold
        )
        print(f"[{res['scene_id']}] Predicted {res['positive_pixels']} disturbance pixels ({res['positive_percentage']:.2f}%)")
        results.append(res)
    return results
