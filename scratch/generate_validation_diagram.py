"""
Generate publication-quality validation diagrams for TECR-Net and baseline models.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# Data from user
data = [
    {"Model": "TECR-Net (Proposed)", "AP": 0.4891, "ROC_AUC": 0.6854, "F1": 0.5315, "P_at_50": 0.71, "Type": "Neural Residual"},
    {"Model": "LightGBM", "AP": 0.4790, "ROC_AUC": 0.6847, "F1": 0.5286, "P_at_50": 0.63, "Type": "GBDT"},
    {"Model": "CatBoost", "AP": 0.4775, "ROC_AUC": 0.6838, "F1": 0.5326, "P_at_50": 0.65, "Type": "GBDT"},
    {"Model": "XGBoost", "AP": 0.4754, "ROC_AUC": 0.6847, "F1": 0.5380, "P_at_50": 0.57, "Type": "GBDT"},
    {"Model": "HistGradientBoosting", "AP": 0.4642, "ROC_AUC": 0.6755, "F1": 0.5196, "P_at_50": 0.58, "Type": "GBDT"},
    {"Model": "Persistence Baseline", "AP": 0.3776, "ROC_AUC": 0.6071, "F1": 0.4644, "P_at_50": 0.46, "Type": "Baseline"}
]
df = pd.DataFrame(data)

# Directory setup
figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)
artifact_dir = Path(r"C:\Users\Alpha08\.gemini\antigravity-ide\brain\61f47c23-8f78-41d9-9bc1-e291b4d48fa6")

# Global style config
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

# Colors: Distinct, academic palette
colors = {
    "TECR-Net (Proposed)": "#1f77b4",       # Deep Royal Blue
    "LightGBM": "#2ca02c",                  # Green
    "CatBoost": "#ff7f0e",                  # Orange
    "XGBoost": "#9467bd",                   # Purple
    "HistGradientBoosting": "#8c564b",      # Brown
    "Persistence Baseline": "#7f7f7f"       # Grey
}

# ==========================================
# Figure 1: 4-Panel Comprehensive Validation Diagram
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
plt.subplots_adjust(wspace=0.28, hspace=0.32)

# Panel A: AP vs ROC-AUC Scatter / Frontier
ax_a = axs[0, 0]
for idx, row in df.iterrows():
    m = row['Model']
    c = colors[m]
    size = 220 if "TECR" in m else 130
    marker = '*' if "TECR" in m else ('o' if "Baseline" not in m else 's')
    ax_a.scatter(row['AP'], row['ROC_AUC'], color=c, s=size, marker=marker, edgecolors='black', linewidth=1.2, zorder=4, label=m)
    
    # Text offsets to avoid overlapping
    offset_x, offset_y = 0.002, 0.002
    if m == "CatBoost":
        offset_y = -0.004
        offset_x = 0.001
    elif m == "LightGBM":
        offset_y = 0.0025
        offset_x = -0.012
    elif m == "XGBoost":
        offset_y = -0.003
        offset_x = 0.002
    elif m == "TECR-Net (Proposed)":
        offset_y = 0.0025
        offset_x = -0.015
    ax_a.annotate(f"{m}\n({row['AP']:.4f}, {row['ROC_AUC']:.4f})", 
                  (row['AP'], row['ROC_AUC']),
                  xytext=(row['AP'] + offset_x, row['ROC_AUC'] + offset_y),
                  fontsize=9, fontweight='bold' if "TECR" in m else 'normal',
                  color='#111111', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

ax_a.set_title("(a) Discrimination & Ranking Frontier (AP vs. ROC-AUC)", fontsize=13, fontweight='bold', pad=10)
ax_a.set_xlabel("Average Precision (AP / PR-AUC)", fontsize=11, fontweight='bold')
ax_a.set_ylabel("ROC-AUC", fontsize=11, fontweight='bold')
ax_a.grid(True, linestyle='--', alpha=0.45)
ax_a.set_xlim(0.36, 0.505)
ax_a.set_ylim(0.59, 0.70)

# Panel B: High-Priority Operational Triage (Precision@50)
ax_b = axs[0, 1]
models_rev = df['Model'].tolist()[::-1]
p50_rev = df['P_at_50'].tolist()[::-1]
bar_colors = [colors[m] for m in models_rev]

y_pos = np.arange(len(models_rev))
bars = ax_b.barh(y_pos, p50_rev, color=bar_colors, height=0.6, edgecolor='black', linewidth=0.8, alpha=0.88)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(models_rev, fontsize=10, fontweight='bold')
ax_b.set_xlabel("Precision@50 (Top 50 Platform Triage)", fontsize=11, fontweight='bold')
ax_b.set_title("(b) Operational Triage Precision (Top 50 Highest Risk)", fontsize=13, fontweight='bold', pad=10)
ax_b.set_xlim(0, 0.85)
ax_b.grid(True, axis='x', linestyle='--', alpha=0.45)

# Value annotations on bars
for bar, val, m in zip(bars, p50_rev, models_rev):
    fw = 'bold' if "TECR" in m else 'normal'
    ax_b.text(val + 0.015, bar.get_y() + bar.get_height()/2, f"{val:.1%} ({val:.2f})", 
              va='center', fontsize=9.5, fontweight=fw, color='#111111')

# Reference threshold for baseline
ax_b.axvline(0.46, color='#7f7f7f', linestyle=':', linewidth=1.4, alpha=0.75, label='Persistence Baseline (46.0%)')
ax_b.legend(loc='lower right', frameon=True, fontsize=9.5)

# Panel C: Grouped Bar Chart Across 4 Core Metrics
ax_c = axs[1, 0]
metrics = ["AP", "ROC_AUC", "F1", "P_at_50"]
metric_labels = ["AP", "ROC-AUC", "F1 Score", "Precision@50"]
n_metrics = len(metrics)
n_models = len(df)
bar_width = 0.13
x = np.arange(n_metrics)

for i, row in df.iterrows():
    m = row['Model']
    vals = [row[met] for met in metrics]
    pos = x + (i - n_models/2 + 0.5) * bar_width
    alpha = 0.95 if "TECR" in m else 0.75
    edge = 'black' if "TECR" in m else 'none'
    lw = 1.2 if "TECR" in m else 0
    ax_c.bar(pos, vals, bar_width, label=m, color=colors[m], alpha=alpha, edgecolor=edge, linewidth=lw)

ax_c.set_xticks(x)
ax_c.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
ax_c.set_ylabel("Metric Value", fontsize=11, fontweight='bold')
ax_c.set_title("(c) Multi-Metric Benchmark Comparison", fontsize=13, fontweight='bold', pad=10)
ax_c.set_ylim(0.30, 0.78)
ax_c.grid(True, axis='y', linestyle='--', alpha=0.45)
ax_c.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), ncol=2, frameon=True, fontsize=8.5)

# Panel D: Relative Advantage over Persistence Baseline (Radar / Spider Chart)
ax_d = fig.add_subplot(2, 2, 4, polar=True)
axs[1, 1].remove() # replace rectangular subplot with polar

categories = ['Average Precision', 'ROC-AUC', 'F1 Score', 'Precision@50']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax_d.set_theta_offset(np.pi / 2)
ax_d.set_theta_direction(-1)
ax_d.set_xticks(angles[:-1])
ax_d.set_xticklabels(categories, fontsize=9.5, fontweight='bold')

for idx, row in df.iterrows():
    m = row['Model']
    values = [row['AP'], row['ROC_AUC'], row['F1'], row['P_at_50']]
    values += values[:1]
    
    lw = 2.5 if "TECR" in m else 1.3
    ls = '-' if "TECR" in m else ('--' if "Persistence" in m else '-.')
    alpha = 0.9 if "TECR" in m else 0.55
    ax_d.plot(angles, values, linewidth=lw, linestyle=ls, label=m, color=colors[m], alpha=alpha)
    if "TECR" in m:
        ax_d.fill(angles, values, color=colors[m], alpha=0.15)

ax_d.set_ylim(0.35, 0.75)
ax_d.set_title("(d) Comparative Model Radar Profile", fontsize=13, fontweight='bold', pad=18)
ax_d.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), frameon=True, fontsize=8.5)

plt.suptitle("Validation Performance Benchmark: TECR-Net vs. Supervised Baselines", 
             fontsize=16, fontweight='bold', y=0.99)

out_fig1 = figures_dir / "validation_performance_diagram.png"
fig.savefig(out_fig1, bbox_inches='tight')
plt.close(fig)

# Copy to brain artifact directory for markdown embedding
shutil.copy(out_fig1, artifact_dir / "validation_performance_diagram.png")

print(f"Validation diagram created successfully at {out_fig1}")
print(f"Copied to artifact dir: {artifact_dir / 'validation_performance_diagram.png'}")
