# WaveAttn SOPSeg: High-Precision Mangrove Canopy Disturbance Mapping via Frequency-Aware Prompted Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain](https://img.shields.io/badge/Domain-Remote%20Sensing%20%7C%20Earth%20Observation-blue.svg)]()
[![Study Area](https://img.shields.io/badge/Location-C%E1%BA%A7n%20Gi%E1%BB%9D%20Biosphere%20Reserve%2C%20Vietnam-brightgreen.svg)]()

Official PyTorch implementation of **WaveAttn SOPSeg** for high-precision, boundary-preserving mapping of localized mangrove canopy gaps and storm-induced disturbance corridors ("vùng gãy đổ") in coastal wetland ecosystems using multi-temporal satellite imagery.

---

## 📌 Abstract & Research Highlights

Mangrove ecosystems provide vital blue carbon sequestration and coastal protection, yet they face severe canopy mortality from extreme storms, lightning, and tidal scouring. In optical satellite observations (e.g., 30 m Landsat archives spanning 2006–2014 in the UNESCO Can Gio Mangrove Biosphere Reserve), mapping localized canopy gaps presents severe challenges:
1. **Extreme Class Imbalance**: Canopy gaps account for $<0.6\%$ of the landscape.
2. **Spectral Mimicry**: Exposed wet mudflats, shallow tidal creeks, and moisture-depleted deadwood share overlapping visible signatures.
3. **Spatial Feature Smearing**: Standard vision transformer (ViT) foundation models downsample imagery by $16\times$, causing micro-disturbances ($<5$ pixels / $<150$ m) to collapse into single tokens, while convolutional decoders over-smooth jagged boundary transitions.

**WaveAttn SOPSeg** addresses these bottlenecks through four synergistic mechanisms:
- **Region-Adaptive Magnification (RAM)** & **Oriented Prompt Mechanism (OPM)**: Dynamically zooms into disturbance candidates and parameterizes rotated bounding prompts to strip away surrounding non-forest mudflats.
- **WaveFusion Multi-Scale Progressive Refinement**: Uses pure PyTorch 2D Haar Discrete Wavelet Transform (DWT/IDWT) to decouple high-frequency directional boundary gradients (LH, HL, HH) from low-frequency regional semantics (LL).
- **Global-Local Contextual Attention (GLCAM)**: Bridges contextual gap relationships across clustered blowdowns while enforcing local spatial constraints.
- **Biophysical Canopy Guidance**: Integrates multi-spectral Normalized Burn Ratio (NBR) directly into the guidance stream to leverage moisture absorption contrasts between deadwood and wet mudflats.

---

## 🔬 Architectural Framework

```
                          +-------------------------------------------------+
                          |   Multi-Spectral Landsat Scene [Bands 1-7]      |
                          +-------------------------------------------------+
                                      |                             |
                                      v                             v
                       [NIR, Red, Green] Composite      [NIR, Red, Green, NBR]
                                      |                             |
                                      v                             v
                     +---------------------------------+   +------------------+
                     |  Region-Adaptive Magnification  |   |   Biophysical    |
                     |     (RAM Windowing, Sin=256)    |   |  Canopy Guidance |
                     +---------------------------------+   +------------------+
                                      |                             |
                                      v                             |
                     +---------------------------------+            |
                     |    Oriented Prompt Mechanism    |            |
                     |  (Rotated Box Prompting [OPM])  |            |
                     +---------------------------------+            |
                                      |                             |
               +----------------------+----------------------+      |
               |                                             |      |
               v                                             v      |
    +--------------------+                        +--------------------+
    | Early ViT-B Block  |                        |  SAM ViT-B Encoder |
    |  Shallow Token Fs  |                        |  (Rank-8 LoRA PEFT)|
    +--------------------+                        +--------------------+
               |                                             |
               |                                             v
               |                                  +--------------------+
               |                                  | SOPSeg Mask Decoder|
               |                                  | Initial Preds [M,E]|
               |                                  +--------------------+
               |                                             |
               +----------------------+----------------------+
                                      |
                                      v
           +--------------------------------------------------------+
           |       Progressive WaveFusion PRM with GLCAM Gates      |
           |--------------------------------------------------------|
           | Stage 1/8 -> 1/4: Haar 2D DWT -> LH/HL/HH -> IDWT     |
           | Stage 1/4 -> 1/2: High-Frequency Gradient Restoration  |
           | Stage 1/2 -> 1/1: Sub-Pixel Boundary Synthesis         |
           +--------------------------------------------------------+
                                      |
                                      v
           +--------------------------------------------------------+
           | Final High-Precision Disturbance Mask & Probability   |
           +--------------------------------------------------------+
```

---

## 📊 Key Manuscript Results

### Table 1: Master Model Benchmark Comparison
Quantitative performance comparison across all deep learning architectures on Cut Chips ($124 \times 124$) and Full Radiometric Scenes:

| Model Architecture | Data Source | IoU | Dice / F1 | Precision | Recall | Boundary IoU | Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Fully-Supervised U-Net (ResNet-34) | Cut Chips | 0.3850 | 0.5559 | 0.4299 | 0.7865 | 0.2810 | 0.9926 |
| Fully-Supervised U-Net (ResNet-34) | Full Scenes | 0.3782 | 0.5488 | 0.4236 | 0.7790 | 0.2740 | 0.9926 |
| Fully-Supervised U-Net (ResNet-50) | Cut Chips | 0.4444 | 0.6154 | 0.6243 | 0.6067 | 0.3120 | 0.9956 |
| Fully-Supervised U-Net (ResNet-50) | Full Scenes | 0.0801 | 0.1484 | 0.0867 | 0.5131 | 0.0984 | 0.9659 |
| SAM + LoRA (Interactive) | Cut Chips | 0.5732 | 0.7287 | 0.8916 | 0.6161 | 0.4350 | 0.9972 |
| SAM + LoRA (Interactive) | Full Scenes | 0.5413 | 0.7024 | 0.8980 | 0.5768 | 0.4180 | 0.9972 |
| SOPSeg (Frozen ViT) | Cut Chips | **0.6980** | **0.8222** | 0.7363 | 0.9307 | **0.9619** | 0.9975 |
| SOPSeg (Frozen ViT) | Full Scenes | 0.7378 | 0.8492 | 0.7755 | **0.9382** | 0.9641 | 0.9981 |
| SOPSeg + LoRA | Cut Chips | 0.5872 | 0.7399 | 0.6506 | 0.8577 | 0.9148 | 0.9965 |
| SOPSeg + LoRA | Full Scenes | 0.7474 | 0.8554 | 0.7914 | 0.9307 | **0.9654** | 0.9982 |
| **WaveAttn SOPSeg [Ours]** | Cut Chips | 0.6468 | 0.7855 | 0.6785 | **0.9326** | 0.9419 | 0.9971 |
| **WaveAttn SOPSeg [Ours]** | Full Scenes | **0.7576** | **0.8621** | **0.7987** | 0.9363 | 0.9634 | **0.9983** |

*Note: On pre-cut chips, standard SOPSeg with Frozen ViT leads in IoU (0.6980 vs. 0.6468) due to the absence of unmasked background mudflats; under operational uncropped Full Scenes, WaveAttn SOPSeg achieves the highest overall accuracy (0.7576 IoU, 0.8621 Dice).*

### Table 2 & 3: Size-Stratified Granular Performance Analysis
Efficacy comparison across canopy disturbance dimensions ($<5$ pixels, $5-15$ pixels, $>15$ pixels):

| Data Source | Instance Scale Tier | Count | Baseline SOPSeg (IoU / BIoU) | WaveAttn SOPSeg (IoU / BIoU) | Delta ($\Delta$ IoU / $\Delta$ BIoU) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Full Scenes** | Small ($<5$ px / $<150$ m) | 42 | 0.6036 / 0.6530 | **0.6335 / 0.6923** | **+2.99 pp / +3.93 pp** |
| **Full Scenes** | Medium ($5-15$ px / $150-450$ m) | 14 | 0.6034 / 0.5446 | **0.6108 / 0.5597** | **+0.74 pp / +1.51 pp** |
| **Full Scenes** | Large ($>15$ px / $>450$ m) | 2 | 0.7589 / 0.6686 | **0.7612 / 0.6482** | **+0.23 pp / -2.04 pp** |
| Cut Chips | Small ($<5$ px) | 42 | 0.5083 / 0.5251 | 0.4912 / 0.5235 | -1.71 pp / -0.16 pp |
| Cut Chips | Medium ($5-15$ px) | 14 | 0.5210 / 0.3880 | **0.5436 / 0.4565** | **+2.26 pp / +6.85 pp** |
| Cut Chips | Large ($>15$ px) | 2 | 0.6191 / 0.4504 | **0.6821 / 0.5479** | **+6.30 pp / +9.75 pp** |

### Table 4: Cumulative Component Ablation Study (Full Scenes)
Stepwise architectural breakdown starting from the fine-tuned SAM baseline:

| Stage / Configuration | IoU | Dice / F1 | Precision | Recall | Boundary IoU (BIoU) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Stage 0: Base (SAM Finetuned) | 0.5413 | 0.7024 | **0.8980** | 0.5768 | 0.4180 |
| Stage 1: + RAM (Horizontal Box Prompt) | 0.1412 | 0.2475 | 0.5230 | 0.1610 | 0.6240 |
| Stage 2: + RAM + OPM (Oriented Prompt) | 0.5842 | 0.7376 | 0.6410 | 0.8415 | 0.8872 |
| Stage 3: SOPSeg (Frozen ViT + Full PRM) | 0.7378 | 0.8492 | 0.7755 | **0.9382** | 0.9641 |
| Stage 4: SOPSeg + LoRA | 0.7474 | 0.8554 | 0.7914 | 0.9307 | **0.9654** |
| **Stage 5: WaveAttn SOPSeg (+ WaveFusion + GLCAM + NBR) [Ours]** | **0.7576** | **0.8621** | 0.7987 | 0.9363 | 0.9634 |

---

## 📁 Repository Structure

```
├── data/                                 # Multi-temporal satellite imagery & reference masks
│   ├── Cut_xxxx/                         # Localized 124x124 pixel image chips
│   └── Full_Scenes_xxxx/                 # Full uncropped radiometric scenes
├── src/                                  # Core library & model architectures
│   ├── __init__.py                       # Package exports (WaveAttnSOPSegModel, etc.)
│   ├── config.py                         # Paths, geographic bounds (EPSG:32648)
│   ├── dataset.py                        # Multi-spectral Landsat loader & connected components
│   ├── indices.py                        # Biophysical index extraction (NDVI, NBR, NDWI)
│   ├── models.py                         # Fully-supervised U-Net & deep CNN baselines
│   ├── sam_satellite.py                  # SAM LoRA adapter (Hu et al. / Scribano et al.)
│   ├── sopseg.py                         # Standard SOPSeg (RAM, OPM, PRM)
│   ├── sopseg_enhanced.py                # WaveAttn SOPSeg (Haar DWT/IDWT, GLCAM, NBR Guidance)
│   └── evaluate.py                       # IoU, Boundary IoU (Cheng et al.), Dice, Precision, Recall
├── scripts/                              # Executable experimentation & reproduction scripts
│   ├── reproduce_tables.py               # Generates and validates Tables 1, 2, 3, 4
│   ├── reproduce_figures.py              # Re-renders Figures 2, 3, 4, 5, 6
│   ├── train_waveattn_sopseg.py          # Standalone training script with CLI flags
│   ├── evaluate_models.py                # Standalone evaluation & GeoTIFF metric assessment
│   ├── export_manuscript_docx.py         # Full manuscript document generator (.docx)
│   ├── plot_waveattn_sopseg_framework.py # Generates Figure 2
│   ├── plot_table1_benchmark.py          # Generates Figure 3
│   ├── plot_figure4_qualitative.py       # Generates Figure 4
│   ├── plot_anhgoc_ablation.py           # Generates Figure 5 (Full Scene ablation curves)
│   └── plot_waveattn_sopseg_evaluation_lc08.py # Generates Figure 6
├── results/                              # Checkpoints, figures, and exported predictions
│   ├── models/                           # Pre-trained weights (.pth)
│   ├── plots/                            # High-resolution figures (.png)
│   └── predictions/                      # Georeferenced prediction masks (.tif)
├── reproduce.py                          # Master single-command reproduction entrypoint
├── requirements.txt                      # Python dependencies
├── setup.py                              # Package installation script
├── pyproject.toml                        # PEP 518 build specification
└── LICENSE                               # MIT License
```

---

## ⚡ Quick Start

### 1. Installation

Clone this repository and set up a Python virtual environment:
```bash
git clone https://github.com/your-username/waveattn-sopseg.git
cd waveattn-sopseg

# Create and activate environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# .\venv\Scripts\activate      # On Windows

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Download Pre-trained Weights

Ensure the official Meta AI SAM ViT-B checkpoint is placed in `results/models/`:
```bash
mkdir -p results/models
wget -O results/models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## 🔁 Reproducing Manuscript Results

You can reproduce all tables, figures, and validation evaluations using the master reproduction script:

```bash
# 1. Reproduce and print all manuscript tables (Tables 1, 2, 3, 4)
python reproduce.py --tables

# 2. Re-render all publication figures (Figures 2, 3, 4, 5, 6)
python reproduce.py --figures

# 3. Evaluate predictions on held-out validation scenes
python reproduce.py --eval

# 4. End-to-end complete reproduction
python reproduce.py --all
```

---

## 🚀 Training & Evaluation

### Training WaveAttn SOPSeg

To train WaveAttn SOPSeg from scratch or fine-tune on full scenes or cut chips:
```bash
# Train on operational Full Radiometric Scenes
python scripts/train_waveattn_sopseg.py --source full --epochs 15 --batch_size 4 --lr 1e-4

# Train on Cut Chips
python scripts/train_waveattn_sopseg.py --source cut --epochs 15 --batch_size 4 --lr 1e-4

# Train on both data regimes sequentially
python scripts/train_waveattn_sopseg.py --source both --epochs 15
```

### Evaluating Pre-trained Checkpoints

To evaluate any model across validation scenes and print detailed confusion metrics:
```bash
# Evaluate WaveAttn SOPSeg on both domains
python scripts/evaluate_models.py --model waveattn_sopseg --source both

# Evaluate standard SOPSeg
python scripts/evaluate_models.py --model sopseg --source both
```

---

## 📄 Citation

If you find this code or our paper useful in your research, please cite:

```bibtex
@article{cangio_mangrove_disturbance_2026,
  title={High-Precision Mangrove Canopy Disturbance Mapping in the Can Gio Biosphere Reserve via Frequency-Aware Prompted Foundation Models},
  author={Hai, N. T. and collaborators},
  journal={Remote Sensing of Environment},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
