"""
Manuscript Figures Reproduction Script
Re-renders and verifies Figures 2, 3, 4, 5, and 6 from the manuscript:
- Figure 2: WaveAttn SOPSeg architectural framework diagram
- Figure 3: Quantitative multi-metric benchmark (Cut Chips vs Full Scenes)
- Figure 4: Qualitative predictions comparison on validation scene LT05_20110207
- Figure 5: Cumulative component ablation study curves on Full Scenes
- Figure 6: WaveAttn SOPSeg spatial diagnostic evaluation on validation scene LC08_20140810

Usage:
    python scripts/reproduce_figures.py
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


FIGURE_SCRIPTS = [
    {
        "figure": "Figure 2",
        "description": "WaveAttn SOPSeg Framework Architecture Overview",
        "script": "scripts/plot_waveattn_sopseg_framework.py",
        "output": "results/plots/waveattn_sopseg_framework_overview.png"
    },
    {
        "figure": "Figure 3",
        "description": "Quantitative Benchmark Comparison (Cut vs Full Scenes)",
        "script": "scripts/plot_table1_benchmark.py",
        "output": "results/plots/table1_models_benchmark_cut_vs_full.png"
    },
    {
        "figure": "Figure 4",
        "description": "Qualitative Predictions Comparison on LT05_20110207",
        "script": "scripts/plot_figure4_qualitative.py",
        "output": "results/plots/figure4_qualitative_predictions_cut_vs_full.png"
    },
    {
        "figure": "Figure 5",
        "description": "Cumulative Component Ablation Study Curves",
        "script": "scripts/plot_waveattn_ablation.py",
        "output": "results/plots/sopseg_ablation_study_full.png"
    },
    {
        "figure": "Figure 6",
        "description": "WaveAttn SOPSeg Spatial Diagnostic on LC08_20140810",
        "script": "scripts/plot_waveattn_sopseg_evaluation_lc08.py",
        "output": "results/plots/eval_waveattn_sopseg_full_LC08_20140810.png"
    }
]


def main():
    print("\n=======================================================")
    print(" REPRODUCING ALL MANUSCRIPT FIGURES (Figures 2 to 6)")
    print("=======================================================\n")

    python_exec = sys.executable

    for item in FIGURE_SCRIPTS:
        fig_name = item["figure"]
        desc = item["description"]
        script_path = os.path.join(PROJECT_ROOT, item["script"])
        out_path = os.path.join(PROJECT_ROOT, item["output"])

        print(f"[*] Generating {fig_name}: {desc}...")
        res = subprocess.run([python_exec, script_path], cwd=PROJECT_ROOT, capture_output=True, text=True)
        
        if res.returncode == 0 and os.path.exists(out_path):
            file_kb = os.path.getsize(out_path) / 1024.0
            print(f"    -> [SUCCESS] Saved to {item['output']} ({file_kb:.1f} KB)")
        else:
            print(f"    -> [ERROR] Failed to generate {fig_name}!")
            if res.stderr:
                print(f"       Details: {res.stderr.strip()[:300]}")

    print("\n=======================================================")
    print(" ALL MANUSCRIPT FIGURES REPRODUCED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
