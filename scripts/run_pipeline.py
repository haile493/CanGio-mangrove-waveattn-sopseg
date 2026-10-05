import os
import sys
import argparse
import torch

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import Config
from src.dataset import discover_scenes, CanGioDataset
from src.models import get_model_from_config
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_all_scenes
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Can Gio Satellite Image Disturbance Segmentation Pipeline")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "deeplabv3plus", "fpn"], help="Architecture")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder")
    parser.add_argument("--source", type=str, default="anhgoc", choices=["anhgoc", "cut", "auto"], help="Imagery source: 'anhgoc' (original scenes) or 'cut' (pre-cut chips)")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--evaluate", action="store_true", help="Run validation and generate plot figures")
    parser.add_argument("--predict", action="store_true", help="Run inference on all scenes and export GeoTIFFs")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline: train -> evaluate -> predict")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # If no action specified, default to --all
    if not (args.train or args.evaluate or args.predict or args.all):
        args.all = True

    config = Config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        arch=args.arch,
        encoder_name=args.encoder,
        data_source=args.source
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Can Gio Satellite Image Disturbance Segmentation")
    print(f"Device: {device} | Architecture: {config.arch} ({config.encoder_name})")
    print(f"Data source: {config.data_source}")
    print(f"Input channels: {config.in_channels} (Spectral: {config.spectral_bands}, Indices: {config.indices})")
    print("=" * 60)

    model = None

    # 1. Training Step
    if args.train or args.all:
        print("\n>>> STEP 1: MODEL TRAINING <<<")
        model, history = train_model(config)

    # 2. Evaluation Step
    if args.evaluate or args.all:
        print("\n>>> STEP 2: MODEL EVALUATION & VISUALIZATION <<<")
        if model is None:
            model = get_model_from_config(config).to(device)
            best_ckpt = os.path.join(config.models_dir, f"best_{config.arch}_{config.encoder_name}.pth")
            if os.path.exists(best_ckpt):
                ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"Loaded best checkpoint: {best_ckpt}")
            else:
                print(f"No checkpoint found at {best_ckpt}. Using random initialization.")

        scenes = discover_scenes(config.data_dir, source=config.data_source)
        val_scenes = [s for s in scenes if any(vy in s["year_folder"] for vy in config.val_years) and s.get("groundtruth_path")]
        if not val_scenes:
            val_scenes = [s for s in scenes if s.get("groundtruth_path")]
            
        val_dataset = CanGioDataset(
            scenes=val_scenes,
            spectral_bands=config.spectral_bands,
            include_indices=config.include_indices,
            indices=config.indices,
            augment=False,
            require_gt=True,
            study_bounds=config.study_bounds
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        print(f"Evaluating across {len(val_scenes)} validation scenes...")
        global_metrics, per_scene = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            threshold=config.threshold,
            save_plots_dir=config.plots_dir
        )
        print("\n--- Validation Performance Metrics ---")
        print(f"Mean IoU:         {global_metrics['iou']:.4f}")
        print(f"Dice (F1) Score:  {global_metrics['dice']:.4f}")
        print(f"Precision:        {global_metrics['precision']:.4f}")
        print(f"Recall:           {global_metrics['recall']:.4f}")
        print(f"Overall Accuracy: {global_metrics['accuracy']:.4f}")
        print(f"Confusion Matrix: TP={global_metrics['tp']}, FP={global_metrics['fp']}, FN={global_metrics['fn']}, TN={global_metrics['tn']}")
        print(f"Evaluation visualization plots saved to: {config.plots_dir}")

    # 3. Prediction Step (Inference & GeoTIFF Export)
    if args.predict or args.all:
        print("\n>>> STEP 3: FULL DATASET INFERENCE & GEOTIFF EXPORT <<<")
        if model is None:
            model = get_model_from_config(config).to(device)
            best_ckpt = os.path.join(config.models_dir, f"best_{config.arch}_{config.encoder_name}.pth")
            if os.path.exists(best_ckpt):
                ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"Loaded best checkpoint: {best_ckpt}")

        pred_results = predict_all_scenes(
            model=model,
            config=config,
            device=device,
            save_dir=config.predictions_dir,
            threshold=config.threshold
        )
        print(f"Exported {len(pred_results)} georeferenced GeoTIFF predictions to: {config.predictions_dir}")

    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()
