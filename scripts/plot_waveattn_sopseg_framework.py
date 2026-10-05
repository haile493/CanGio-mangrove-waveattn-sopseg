import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, ConnectionStyle

def draw_framework_overview(output_path: str):
    # Setup canvas
    fig = plt.figure(figsize=(18, 10.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Color Palette (Nature / IEEE TGRS publication palette)
    C_BG_BOX = "#F8FAFC"
    C_BORDER = "#94A3B8"
    
    # Module Themes
    C_INPUT = {"bg": "#E0F2FE", "border": "#0284C7", "txt": "#0369A1"}
    C_PROMPT = {"bg": "#FFE4E6", "border": "#E11D48", "txt": "#BE123C"}
    C_BACKBONE = {"bg": "#FEF3C7", "border": "#D97706", "txt": "#B45309"}
    C_LORA = {"bg": "#FEF08A", "border": "#CA8A04", "txt": "#854D0E"}
    C_GLCAM = {"bg": "#F3E8FF", "border": "#9333EA", "txt": "#6B21A8"}
    C_WAVE = {"bg": "#DCFCE7", "border": "#16A34A", "txt": "#15803D"}
    C_PRM = {"bg": "#EFF6FF", "border": "#3B82F6", "txt": "#1D4ED8"}
    C_OUT = {"bg": "#FEE2E2", "border": "#DC2626", "txt": "#991B1B"}

    # Helper: draw fancy box
    def draw_box(x, y, w, h, title, subtitle="", theme=C_INPUT, radius=1.2, lw=1.5, ls="-", title_fs=10, sub_fs=8):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            ec=theme["border"], fc=theme["bg"],
            lw=lw, ls=ls, zorder=2
        )
        ax.add_patch(box)
        if title:
            ax.text(
                x + w/2, y + h/2 + (1.2 if subtitle else 0),
                title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=theme["txt"],
                zorder=3
            )
        if subtitle:
            ax.text(
                x + w/2, y + h/2 - 1.5,
                subtitle, ha="center", va="center",
                fontsize=sub_fs, fontweight="normal", color="#334155",
                zorder=3
            )
        return box

    # Helper: draw connection arrow
    def draw_arrow(start, end, color="#475569", lw=1.6, style="-|>", rad=0.0):
        con = patches.FancyArrowPatch(
            start, end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style,
            color=color,
            lw=lw,
            mutation_scale=12,
            zorder=4
        )
        ax.add_patch(con)

    # =========================================================================
    # SECTION CONTAINERS (Big functional dashed blocks)
    # =========================================================================
    # 1. Input & Adaptive Prompting
    box1 = FancyBboxPatch((1.5, 52), 22, 42, boxstyle="round,pad=0.5,rounding_size=2",
                          ec="#CBD5E1", fc="#F8FAFC", lw=1.5, ls="--", zorder=1)
    ax.add_patch(box1)
    ax.text(12.5, 91.5, "1. Input & Oriented Prompting", fontsize=11, fontweight="bold", color="#1E293B", ha="center")

    # 2. Foundation Encoder with LoRA
    box2 = FancyBboxPatch((25.5, 52), 23, 42, boxstyle="round,pad=0.5,rounding_size=2",
                          ec="#CBD5E1", fc="#F8FAFC", lw=1.5, ls="--", zorder=1)
    ax.add_patch(box2)
    ax.text(37, 91.5, "2. SAM ViT Encoder + LoRA", fontsize=11, fontweight="bold", color="#1E293B", ha="center")

    # 3. WaveFusion PRM Multi-Scale Refinement
    box3 = FancyBboxPatch((50.5, 43), 48, 51, boxstyle="round,pad=0.5,rounding_size=2",
                          ec="#CBD5E1", fc="#F8FAFC", lw=1.5, ls="--", zorder=1)
    ax.add_patch(box3)
    ax.text(74.5, 91.5, "3. WaveFusion PRM with Biophysical NBR Guidance", fontsize=11, fontweight="bold", color="#1E293B", ha="center")

    # 4. Inset Details (Bottom Container)
    box4 = FancyBboxPatch((1.5, 3), 97, 36, boxstyle="round,pad=0.5,rounding_size=2",
                          ec="#94A3B8", fc="#F1F5F9", lw=1.5, ls="-", zorder=1)
    ax.add_patch(box4)
    ax.text(50, 36.5, "Detailed Architectural Submodules: WaveFusion Block & GLCAM Attention", fontsize=12, fontweight="bold", color="#0F172A", ha="center")

    # =========================================================================
    # 1. INPUT & ORIENTED PROMPTING MODULES
    # =========================================================================
    draw_box(3, 76, 19, 12, "Landsat Scene (30m)", "6 Bands [B,G,R,NIR,SW1,SW2]\n+ Indices [NDVI, NBR, NDWI]", C_INPUT, title_fs=10, sub_fs=8)
    draw_box(3, 58, 9, 13, "RAM\nModule", "Region-Adaptive\nMagnification\n(256×256 Patch)", C_INPUT, title_fs=9, sub_fs=7.5)
    draw_box(13, 58, 9, 13, "OPM\nModule", "Oriented Prompt\nRotated Box\n(xc, yc, w, h, θ)", C_PROMPT, title_fs=9, sub_fs=7.5)

    draw_arrow((12.5, 76), (7.5, 71), color="#0284C7")
    draw_arrow((12.5, 76), (17.5, 71), color="#E11D48")

    # =========================================================================
    # 2. FOUNDATION ENCODER WITH LORA
    # =========================================================================
    draw_box(27.5, 71, 19, 17, "SAM ViT-B Encoder", "12 Transformer Blocks\nPatch Embed stride 16×\nPosition Interp (256×256)", C_BACKBONE, title_fs=10, sub_fs=8)
    draw_box(29.5, 73, 15, 6, "LoRA Adapters", "Rank r=8 on Wq, Wv (α=16)", C_LORA, title_fs=9, sub_fs=7)

    draw_box(27.5, 55, 19, 11, "SAM Prompt Encoder", "Encodes (xc, yc, w, h, θ)\nPositional Embeddings", C_PROMPT, title_fs=9.5, sub_fs=7.5)

    # Connections from Module 1 to Module 2
    draw_arrow((12, 64.5), (27.5, 78), color="#0284C7")
    draw_arrow((22, 60), (27.5, 60), color="#E11D48")

    # Intermediate tokens
    ax.text(49, 83.5, "Shallow Fs\n(768, 16×16)", fontsize=8, fontweight="bold", color="#7C3AED", ha="center")
    ax.text(49, 73.5, "Deep Fd\n(256, 16×16)", fontsize=8, fontweight="bold", color="#7C3AED", ha="center")
    ax.text(49, 60.5, "Coarse P0\n[M0; E0] (64×64)", fontsize=8, fontweight="bold", color="#D97706", ha="center")

    draw_arrow((46.5, 82), (52, 82), color="#7C3AED")
    draw_arrow((46.5, 75), (52, 75), color="#7C3AED")
    draw_arrow((46.5, 60), (52, 60), color="#D97706")

    # =========================================================================
    # 3. WAVEFUSION PRM REFINEMENT
    # =========================================================================
    # GLCAM Attention Blocks
    draw_box(52.5, 72, 11, 14, "GLCAM", "Global-Local\nContext Attention\n(Fs & Fd Gating)", C_GLCAM, title_fs=9.5, sub_fs=7.5)

    # Biophysical NBR Guidance Injection
    draw_box(52.5, 46, 11, 11, "Biophysical\nGuidance", "4-ch [NIR, R, G, NBR]\nCanopy Moisture", C_INPUT, title_fs=9, sub_fs=7)

    # PRM Multi-Scale Cascade
    # Scale 1/8 -> 1/4
    draw_box(66.5, 74, 9, 13, "Stage 1\n(1/8 → 1/4)", "WaveFusion 1\nFeat: 64×64\nP4 = [M4, E4]", C_PRM, title_fs=8.5, sub_fs=7)
    # Scale 1/4 -> 1/2
    draw_box(77.5, 74, 9, 13, "Stage 2\n(1/4 → 1/2)", "WaveFusion 2\nFeat: 128×128\nP2 = [M2, E2]", C_PRM, title_fs=8.5, sub_fs=7)
    # Scale 1/2 -> 1/1
    draw_box(88.5, 74, 9, 13, "Stage 3\n(1/2 → 1/1)", "WaveFusion 3\nFeat: 256×256\nP1 = [M1, E1]", C_WAVE, title_fs=8.5, sub_fs=7)

    # Arrows in PRM
    draw_arrow((63.5, 79), (66.5, 79), color="#3B82F6")
    draw_arrow((75.5, 79), (77.5, 79), color="#3B82F6")
    draw_arrow((86.5, 79), (88.5, 79), color="#16A34A")

    # NBR guidance injection arrows
    draw_arrow((58, 57), (71, 74), color="#0284C7", rad=-0.1)
    draw_arrow((58, 57), (82, 74), color="#0284C7", rad=-0.15)
    draw_arrow((58, 57), (93, 74), color="#0284C7", rad=-0.2)

    # Multi-task heads & Outputs
    draw_box(66.5, 46, 14, 11, "Multi-Task Heads", "Mask M1 (256×256)\nEdge E1 (Laplacian)\nIoU Score Head", C_OUT, title_fs=9, sub_fs=7.5)
    draw_box(83.5, 46, 14, 11, "Final Output", "Crisp Disturbance\nBoundaries\nIoU: 0.758 | BIoU: 0.963", C_OUT, title_fs=9.5, sub_fs=7.5)

    draw_arrow((93, 74), (73.5, 57), color="#DC2626", rad=-0.2)
    draw_arrow((80.5, 51.5), (83.5, 51.5), color="#DC2626")

    # =========================================================================
    # 4. DETAILED INSET MODULES (BOTTOM HALF)
    # =========================================================================
    # Inset 4A: WaveFusion Refinement Block Architecture
    ax.text(26, 32.5, "(A) WaveFusion Refinement Block (Haar 2D DWT / IDWT)", fontsize=10, fontweight="bold", color="#15803D", ha="center")
    
    # Input feat
    draw_box(3, 17, 7, 12, "Input X\n(C, H, W)", "Residual\nConnection", C_PRM, title_fs=8.5, sub_fs=7)
    
    # Haar DWT
    draw_box(12, 17, 7, 12, "Haar 2D\nDWT", "Group Conv\nStride 2", C_WAVE, title_fs=8.5, sub_fs=7)
    draw_arrow((10, 23), (12, 23), color="#16A34A")

    # Subbands branches
    # Low freq branch
    draw_box(21, 23, 10, 8, "Low-Freq LL", "Semantics & Area\nResBlock Low", C_PRM, title_fs=8, sub_fs=6.5)
    # High freq branch
    draw_box(21, 10, 10, 10, "High-Freq Subbands\n(LH, HL, HH)", "Horizontal, Vertical,\nDiagonal Edges\nConv1x1 + ResBlock", C_WAVE, title_fs=8, sub_fs=6.5)

    draw_arrow((19, 25), (21, 27), color="#3B82F6")
    draw_arrow((19, 21), (21, 15), color="#16A34A")

    # Haar IDWT Reconstruction
    draw_box(33, 17, 7, 12, "Haar 2D\nIDWT", "Transpose Conv\nReconstruction", C_WAVE, title_fs=8.5, sub_fs=7)
    draw_arrow((31, 27), (33, 25), color="#3B82F6")
    draw_arrow((31, 15), (33, 21), color="#16A34A")

    # Output projection + residual addition
    draw_box(42, 17, 7, 12, "Output Xout\n(Cout, H, W)", "Conv3x3 + BN\n+ Shortcut(X)", C_PRM, title_fs=8.5, sub_fs=7)
    draw_arrow((40, 23), (42, 23), color="#16A34A")
    # Residual skip arrow
    draw_arrow((6.5, 29), (45.5, 29), color="#64748B", rad=0.25, style="->")
    ax.text(26, 30.5, "Identity / Conv 1x1 Shortcut", fontsize=7.5, color="#64748B", ha="center")

    # Inset 4B: Global-Local Contextual Attention Module (GLCAM)
    ax.text(74, 32.5, "(B) Global-Local Contextual Attention Module (GLCAM)", fontsize=10, fontweight="bold", color="#6B21A8", ha="center")

    # Input to GLCAM
    draw_box(52, 17, 7, 12, "Input Feat\n(C, H, W)", "Fs / Fd", C_GLCAM, title_fs=8.5, sub_fs=7)

    # Channel attention branch
    draw_box(61, 22.5, 11, 9, "Channel Attention", "AvgPool + MaxPool\n+ Shared MLP (1x1)\nSigmoid Gating (Mc)", C_GLCAM, title_fs=8, sub_fs=6.5)
    draw_arrow((59, 25), (61, 27), color="#9333EA")

    # Spatial attention branch
    draw_box(61, 8.5, 11, 11, "Spatial Attention", "Mean + Max Channel\n+ Conv 7x7 (stride 1)\nSigmoid Gating (Ms)", C_GLCAM, title_fs=8, sub_fs=6.5)
    draw_arrow((59, 21), (61, 14), color="#9333EA")

    # Attention fusion & suppression of mudflats
    draw_box(74, 17, 10, 12, "Context\nGating", "X · Mc · Ms\nSuppresses Tidal Mud\n& Water Noise", C_GLCAM, title_fs=8.5, sub_fs=7)
    draw_arrow((72, 27), (74, 25), color="#9333EA")
    draw_arrow((72, 14), (74, 21), color="#9333EA")

    # Refined output
    draw_box(86, 17, 10, 12, "Spotlighted\nFeatures", "Residual Output\nX + Gated(X)", C_WAVE, title_fs=8.5, sub_fs=7)
    draw_arrow((84, 23), (86, 23), color="#16A34A")
    draw_arrow((55.5, 29), (91, 29), color="#64748B", rad=0.25, style="->")
    ax.text(73.5, 30.5, "Residual Connection (+ X)", fontsize=7.5, color="#64748B", ha="center")

    # Save diagram
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Framework overview diagram successfully generated at: {output_path}")

if __name__ == "__main__":
    out_img = "results/plots/waveattn_sopseg_framework_overview.png"
    draw_framework_overview(out_img)
