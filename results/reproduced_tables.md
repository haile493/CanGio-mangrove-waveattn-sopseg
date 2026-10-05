# Reproduced Manuscript Benchmark Tables

Generated from experimental checkpoints and validation datasets.

### Table 1. Quantitative Performance Comparison across Deep Learning Architectures

| Model Architecture | Data Source | IoU | Dice / F1 | Precision | Recall | Boundary IoU | Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Fully-Supervised U-Net (ResNet-34) | Cut Chips | 0.3850 | 0.5559 | 0.4299 | 0.7865 | 0.2810 | 0.9926 |
| Fully-Supervised U-Net (ResNet-34) | Full Scenes | 0.3782 | 0.5488 | 0.4236 | 0.7790 | 0.2740 | 0.9926 |
| Fully-Supervised U-Net (ResNet-50) | Cut Chips | 0.4444 | 0.6154 | 0.6243 | 0.6067 | 0.3120 | 0.9956 |
| Fully-Supervised U-Net (ResNet-50) | Full Scenes | 0.0801 | 0.1484 | 0.0867 | 0.5131 | 0.0984 | 0.9659 |
| SAM + LoRA (Interactive) | Cut Chips | 0.5732 | 0.7287 | 0.8916 | 0.6161 | 0.4350 | 0.9972 |
| SAM + LoRA (Interactive) | Full Scenes | 0.5413 | 0.7024 | 0.8980 | 0.5768 | 0.4180 | 0.9972 |
| SOPSeg (Frozen ViT) | Cut Chips | 0.6980 | 0.8222 | 0.7363 | 0.9307 | 0.9619 | 0.9975 |
| SOPSeg (Frozen ViT) | Full Scenes | 0.7378 | 0.8492 | 0.7755 | 0.9382 | 0.9641 | 0.9981 |
| SOPSeg + LoRA | Cut Chips | 0.5872 | 0.7399 | 0.6506 | 0.8577 | 0.9148 | 0.9965 |
| SOPSeg + LoRA | Full Scenes | 0.7474 | 0.8554 | 0.7914 | 0.9307 | 0.9654 | 0.9982 |
| WaveAttn SOPSeg [Ours] | Cut Chips | 0.6468 | 0.7855 | 0.6785 | 0.9326 | 0.9419 | 0.9971 |
| WaveAttn SOPSeg [Ours] | Full Scenes | 0.7576 | 0.8621 | 0.7987 | 0.9363 | 0.9634 | 0.9983 |


### Table 2. Baseline SOPSeg Size-Stratified Granular Performance Analysis

| Data Source | Instance Scale Tier | Count | Mean Mask IoU | Mean Boundary IoU |
|:---|:---|:---:|:---:|:---:|
| Cut Chips | Small (<5 pixels) | 42 | 0.5083 | 0.5251 |
| Cut Chips | Medium (5-15 pixels) | 14 | 0.5210 | 0.3880 |
| Cut Chips | Large (>15 pixels) | 2 | 0.6191 | 0.4504 |
| Full Scenes | Small (<5 pixels) | 42 | 0.6036 | 0.6530 |
| Full Scenes | Medium (5-15 pixels) | 14 | 0.6034 | 0.5446 |
| Full Scenes | Large (>15 pixels) | 2 | 0.7589 | 0.6686 |


### Table 3. Proposed WaveAttn SOPSeg Size-Stratified Granular Performance Analysis

| Data Source | Instance Scale Tier | Count | Mean Mask IoU | Mean Boundary IoU |
|:---|:---|:---:|:---:|:---:|
| Cut Chips | Small (<5 pixels) | 42 | 0.4912 | 0.5235 |
| Cut Chips | Medium (5-15 pixels) | 14 | 0.5436 | 0.4565 |
| Cut Chips | Large (>15 pixels) | 2 | 0.6821 | 0.5479 |
| Full Scenes | Small (<5 pixels) | 42 | 0.6335 | 0.6923 |
| Full Scenes | Medium (5-15 pixels) | 14 | 0.6108 | 0.5597 |
| Full Scenes | Large (>15 pixels) | 2 | 0.7612 | 0.6482 |


### Table 4. Cumulative Component Ablation Study on Full Radiometric Scenes

| Model Configuration / Ablation Stage | IoU | Dice / F1 | Precision | Recall | Boundary IoU |
|:---|:---:|:---:|:---:|:---:|:---:|
| Stage 0: Base (SAM Finetuned) | 0.5413 | 0.7024 | 0.8980 | 0.5768 | 0.4180 |
| Stage 1: + RAM (Horizontal Box Prompt) | 0.1412 | 0.2475 | 0.5230 | 0.1610 | 0.6240 |
| Stage 2: + RAM + OPM (Oriented Prompt) | 0.5842 | 0.7376 | 0.6410 | 0.8415 | 0.8872 |
| Stage 3: SOPSeg (Frozen ViT + Full PRM) | 0.7378 | 0.8492 | 0.7755 | 0.9382 | 0.9641 |
| Stage 4: SOPSeg + LoRA | 0.7474 | 0.8554 | 0.7914 | 0.9307 | 0.9654 |
| Stage 5: WaveAttn SOPSeg (+ WaveFusion + GLCAM + NBR) [Ours] | 0.7576 | 0.8621 | 0.7987 | 0.9363 | 0.9634 |

