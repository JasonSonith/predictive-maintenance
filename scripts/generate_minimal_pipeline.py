#!/usr/bin/env python3
"""
Generate minimalistic predictive maintenance pipeline diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Compact figure size
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 8)
ax.set_ylim(0, 10)
ax.axis('off')

# Minimal color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'gray': '#9E9E9E',
    'purple': '#9C27B0',
    'yellow': '#FFC107'
}

def draw_box(ax, x, y, width, height, label, color, fontsize=9):
    """Draw simple rounded box"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle='round,pad=0.03',
                          facecolor=color, edgecolor='white',
                          linewidth=2, alpha=0.85, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color='#555555', width=2, style='-'):
    """Draw simple arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=15,
                           linewidth=width, color=color,
                           linestyle=style, zorder=1)
    ax.add_patch(arrow)

# Vertical positions
y_start = 9.2
y_gap = 1.3

# Stage positions
datasets_y = y_start
prep_y = datasets_y - y_gap
features_y = prep_y - y_gap
train_y = features_y - y_gap
threshold_y = train_y - y_gap
evaluate_y = threshold_y - y_gap
production_y = evaluate_y - y_gap

center_x = 4

# Input Datasets (compact, single row)
dataset_text = "4 Datasets\nIMS | CWRU | AI4I | C-MAPSS"
draw_box(ax, center_x, datasets_y, 6, 0.7, dataset_text, colors['blue'], fontsize=8)

# Arrow from datasets
draw_arrow(ax, center_x, datasets_y - 0.4, center_x, prep_y + 0.35)

# Stage 1: Data Preparation
draw_box(ax, center_x, prep_y, 5, 0.6, "Data Preparation\nprep_data.py", colors['green'])

# Arrow
draw_arrow(ax, center_x, prep_y - 0.35, center_x, features_y + 0.4)

# Stage 2: Feature Engineering
draw_box(ax, center_x, features_y, 5, 0.7,
         "Feature Engineering\nmake_features.py\nWindow: 2048 | Stride: 1024",
         colors['green'], fontsize=8)

# Arrow
draw_arrow(ax, center_x, features_y - 0.4, center_x, train_y + 0.5)

# Stage 3: Model Training (with models listed inside)
draw_box(ax, center_x, train_y, 5, 0.9,
         "Model Training (train.py)\nIsolation Forest • kNN-LOF\nOne-Class SVM • Autoencoder",
         colors['green'], fontsize=8)

# Arrow
draw_arrow(ax, center_x, train_y - 0.5, center_x, threshold_y + 0.35)

# Stage 4: Threshold Calibration
draw_box(ax, center_x, threshold_y, 5, 0.6,
         "Threshold Calibration\nthreshold.py • FAR: 0.2/week",
         colors['green'], fontsize=8)

# Arrow
draw_arrow(ax, center_x, threshold_y - 0.35, center_x, evaluate_y + 0.35)

# Stage 5: Evaluation
draw_box(ax, center_x, evaluate_y, 5, 0.6,
         "Evaluation & Reporting\nevaluate.py • Metrics • SHAP",
         colors['green'], fontsize=8)

# Arrow
draw_arrow(ax, center_x, evaluate_y - 0.35, center_x, production_y + 0.35)

# Stage 6: Production (dashed box)
prod_box = FancyBboxPatch((center_x - 2.5, production_y - 0.3), 5, 0.6,
                          boxstyle='round,pad=0.03',
                          facecolor=colors['purple'], edgecolor='white',
                          linewidth=2, linestyle='--', alpha=0.85, zorder=2)
ax.add_patch(prod_box)
ax.text(center_x, production_y, "Production Scoring\nscore_batch.py",
        ha='center', va='center', fontsize=9, color='white',
        fontweight='bold', zorder=3)

# Config annotation (minimal, on the side)
config_box = FancyBboxPatch((6.5, 4.5), 1.3, 2.5,
                            boxstyle='round,pad=0.05',
                            facecolor=colors['yellow'], edgecolor='white',
                            linewidth=1.5, alpha=0.8, zorder=2)
ax.add_patch(config_box)

ax.text(7.15, 6.6, "Config\nDriven", ha='center', va='center',
        fontsize=8, fontweight='bold', zorder=3)

config_detail = "YAML\nConfigs\n\nDataset\n+\nModel\nSpecs"
ax.text(7.15, 5.3, config_detail, ha='center', va='center',
        fontsize=6.5, zorder=3)

# Dashed arrows from config to stages
for y_pos in [prep_y, features_y, train_y, threshold_y]:
    draw_arrow(ax, 6.5, 5.5, 6.5, y_pos, color=colors['orange'], width=1.2, style='--')

# Input/Output for production
ax.text(1.2, production_y, "New\nData", ha='center', va='center',
        fontsize=7, bbox=dict(boxstyle='round', facecolor=colors['blue'],
                              alpha=0.7, edgecolor='white', linewidth=1.5),
        color='white', fontweight='bold')

ax.text(6.8, production_y, "Alerts", ha='center', va='center',
        fontsize=7, bbox=dict(boxstyle='round', facecolor=colors['gray'],
                              alpha=0.7, edgecolor='white', linewidth=1.5),
        color='white', fontweight='bold')

draw_arrow(ax, 1.8, production_y, center_x - 2.5, production_y, width=1.5)
draw_arrow(ax, center_x + 2.5, production_y, 6.3, production_y, width=1.5)

# Save
output_path = '/mnt/c/Users/Jason/predictive-maintenance/docs/conference-templates/data-mining-report/figures/ml_pipeline_flowchart.pdf'
plt.tight_layout()
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Minimalistic pipeline diagram saved to:\n  {output_path}")
plt.close()
