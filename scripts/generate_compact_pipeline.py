#!/usr/bin/env python3
"""
Generate compact, readable predictive maintenance pipeline architecture diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

# Set up figure with better proportions for academic papers
# Landscape orientation, more compact
fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 12)
ax.set_ylim(0, 14)
ax.axis('off')

# Define colors
colors = {
    'lightblue': '#E8F5FF',
    'darkblue': '#1E88E5',
    'lightgreen': '#EDF7ED',
    'darkgreen': '#43A047',
    'lightorange': '#FFEBCD',
    'darkorange': '#FB8C00',
    'lightgray': '#FAFAFA',
    'darkgray': '#606060',
    'lightyellow': '#FFFDE7',
    'orange': '#FF9800',
    'lightpurple': '#F8EBFA',
    'darkpurple': '#8E24AA'
}

def draw_rounded_box(ax, x, y, width, height, label, sublabel='', details='',
                     facecolor='white', edgecolor='black', linewidth=1.5,
                     fontsize=10, sublabel_fontsize=8):
    """Draw a rounded rectangle with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle='round,pad=0.05',
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=linewidth, zorder=2)
    ax.add_patch(box)

    # Main label
    y_offset = 0.12 if sublabel or details else 0
    ax.text(x, y + y_offset, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', zorder=3)

    # Sublabel
    if sublabel:
        ax.text(x, y - 0.10, sublabel, ha='center', va='center',
                fontsize=sublabel_fontsize, family='monospace', zorder=3)

    # Details
    if details:
        ax.text(x, y - 0.28, details, ha='center', va='center',
                fontsize=7, style='italic', zorder=3)

def draw_cylinder(ax, x, y, width, height, label, facecolor='white', edgecolor='black'):
    """Draw a cylinder shape for data storage"""
    rect = mpatches.Rectangle((x - width/2, y - height/2), width, height,
                               facecolor=facecolor, edgecolor=edgecolor,
                               linewidth=1.2, zorder=2)
    ax.add_patch(rect)

    top_ellipse = Ellipse((x, y + height/2), width, height * 0.3,
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=1.2, zorder=3)
    ax.add_patch(top_ellipse)

    ax.text(x, y, label, ha='center', va='center',
            fontsize=8, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, style='solid', color='#606060', width=1.8):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=18,
                           linewidth=width, color=color,
                           linestyle=style, zorder=1)
    ax.add_patch(arrow)

# Adjusted positions for compact layout
datasets_y = 13.2
prep_y = 11.8
clean_y = 10.7
features_y = 9.6
featurevec_y = 8.5
train_y = 7.0
models_y = 5.5
threshold_y = 4.5
thresholds_y = 3.4
evaluate_y = 2.3
reports_y = 1.2
score_y = 0.4

# Top row - Datasets (4 boxes, more compact spacing)
dataset_names = ['IMS\nBearings', 'CWRU\nBearings', 'AI4I\nMfg', 'C-MAPSS\nTurbofan']
dataset_x_positions = [2.5, 4.3, 6.1, 7.9]

for i, (name, x_pos) in enumerate(zip(dataset_names, dataset_x_positions)):
    draw_rounded_box(ax, x_pos, datasets_y, 1.5, 0.6, name,
                     facecolor=colors['lightblue'], edgecolor=colors['darkblue'],
                     fontsize=9)
    draw_arrow(ax, x_pos, datasets_y - 0.35, x_pos, prep_y + 0.4)

# Stage 1: Data Preparation
draw_rounded_box(ax, 5.2, prep_y, 3.8, 0.75, 'Data Preparation',
                 sublabel='prep_data.py',
                 facecolor=colors['lightgreen'], edgecolor=colors['darkgreen'])

# Config box
draw_rounded_box(ax, 9.2, prep_y, 1.1, 0.45, 'YAML\nConfig',
                 facecolor=colors['lightyellow'], edgecolor=colors['orange'],
                 fontsize=7)

# Clean Data cylinder
draw_cylinder(ax, 5.2, clean_y, 2.2, 0.6, 'Clean Data\n(Parquet/CSV)',
              facecolor=colors['lightgray'], edgecolor=colors['darkgray'])

draw_arrow(ax, 5.2, prep_y - 0.4, 5.2, clean_y + 0.35)

# Stage 2: Feature Engineering
details_text = 'Win: 2048 | Stride: 1024\nmean, std, RMS, kurtosis...'
draw_rounded_box(ax, 5.2, features_y, 3.8, 0.85, 'Feature Engineering',
                 sublabel='make_features.py', details=details_text,
                 facecolor=colors['lightgreen'], edgecolor=colors['darkgreen'])

draw_arrow(ax, 5.2, clean_y - 0.35, 5.2, features_y + 0.45)

# Feature Vectors cylinder
draw_cylinder(ax, 5.2, featurevec_y, 2.2, 0.6, 'Feature Vectors',
              facecolor=colors['lightgray'], edgecolor=colors['darkgray'])

draw_arrow(ax, 5.2, features_y - 0.45, 5.2, featurevec_y + 0.35)

# Stage 3: Model Training (with 4 models inside)
train_box = draw_rounded_box(ax, 5.2, train_y, 3.8, 1.5, '',
                             facecolor=colors['lightgreen'], edgecolor=colors['darkgreen'])

ax.text(5.2, train_y + 0.63, 'Model Training', ha='center', va='center',
        fontsize=10, fontweight='bold', zorder=3)
ax.text(5.2, train_y + 0.38, 'train.py', ha='center', va='center',
        fontsize=8, family='monospace', zorder=3)

# Four models inside (2x2 grid)
model_names = ['Isolation\nForest', 'kNN-LOF', 'One-Class\nSVM', 'Autoenc.']
model_positions = [(4.3, train_y + 0.05), (6.1, train_y + 0.05),
                   (4.3, train_y - 0.35), (6.1, train_y - 0.35)]

for name, (mx, my) in zip(model_names, model_positions):
    draw_rounded_box(ax, mx, my, 1.5, 0.42, name,
                     facecolor=colors['lightorange'], edgecolor=colors['darkorange'],
                     fontsize=7.5)

ax.text(5.2, train_y - 0.68, 'Train on normal data only', ha='center', va='center',
        fontsize=7, style='italic', zorder=3)

draw_arrow(ax, 5.2, featurevec_y - 0.35, 5.2, train_y + 0.78)

# Trained Models cylinder
draw_cylinder(ax, 5.2, models_y, 2.2, 0.6, 'Trained Models\n(.joblib, .pth)',
              facecolor=colors['lightgray'], edgecolor=colors['darkgray'])

draw_arrow(ax, 5.2, train_y - 0.78, 5.2, models_y + 0.35)

# Stage 4: Threshold Calibration
draw_rounded_box(ax, 5.2, threshold_y, 3.8, 0.75, 'Threshold Calibration',
                 sublabel='threshold.py',
                 details='Target FAR: 0.2/week → Threshold',
                 facecolor=colors['lightgreen'], edgecolor=colors['darkgreen'])

draw_arrow(ax, 5.2, models_y - 0.35, 5.2, threshold_y + 0.4)

# Calibrated Thresholds cylinder
draw_cylinder(ax, 5.2, thresholds_y, 2.2, 0.6, 'Calibrated\nThresholds',
              facecolor=colors['lightgray'], edgecolor=colors['darkgray'])

draw_arrow(ax, 5.2, threshold_y - 0.4, 5.2, thresholds_y + 0.35)

# Stage 5: Evaluation & Reporting
draw_rounded_box(ax, 5.2, evaluate_y, 3.8, 0.75, 'Evaluation & Reporting',
                 sublabel='evaluate.py',
                 details='Metrics, SHAP, Visualizations',
                 facecolor=colors['lightgreen'], edgecolor=colors['darkgreen'])

draw_arrow(ax, 5.2, thresholds_y - 0.35, 5.2, evaluate_y + 0.4)

# Reports cylinder
draw_cylinder(ax, 5.2, reports_y, 2.2, 0.6, 'Reports\n(HTML, JSON)',
              facecolor=colors['lightgray'], edgecolor=colors['darkgray'])

draw_arrow(ax, 5.2, evaluate_y - 0.4, 5.2, reports_y + 0.35)

# Stage 6: Production Scoring (dashed border)
prod_box = FancyBboxPatch((3.3, score_y - 0.35), 3.8, 0.7,
                          boxstyle='round,pad=0.05',
                          facecolor=colors['lightpurple'],
                          edgecolor=colors['darkpurple'],
                          linewidth=2, linestyle='--', zorder=2)
ax.add_patch(prod_box)

ax.text(5.2, score_y + 0.12, 'Production Scoring', ha='center', va='center',
        fontsize=10, fontweight='bold', zorder=3)
ax.text(5.2, score_y - 0.12, 'score_batch.py', ha='center', va='center',
        fontsize=8, family='monospace', zorder=3)

draw_arrow(ax, 5.2, reports_y - 0.35, 5.2, score_y + 0.38)

# New Data input
draw_rounded_box(ax, 1.8, score_y, 1.3, 0.5, 'New\nData',
                 facecolor=colors['lightblue'], edgecolor=colors['darkblue'],
                 fontsize=8)
draw_arrow(ax, 2.5, score_y, 3.3, score_y)

# Anomaly Alerts output
draw_cylinder(ax, 9.0, score_y, 1.8, 0.5, 'Anomaly\nAlerts',
              facecolor=colors['lightgray'], edgecolor=colors['darkgray'])
draw_arrow(ax, 7.1, score_y, 8.1, score_y)

# Dashed arrow from models to production
draw_arrow(ax, 4.1, models_y - 0.25, 3.5, score_y + 0.15, style='--', width=1.5)

# Configuration Panel (more compact)
config_panel = FancyBboxPatch((9.8, 9.9), 2.0, 2.0,
                              boxstyle='round,pad=0.05',
                              facecolor=colors['lightyellow'],
                              edgecolor=colors['orange'],
                              linewidth=1.5, zorder=2)
ax.add_patch(config_panel)

ax.text(10.8, 11.6, 'Config-Driven', ha='center', va='center',
        fontsize=9, fontweight='bold', zorder=3)

config_text = '''• Dataset configs
  ims.yaml...
• Model configs
  iforest.yaml...
• Reproducible
  experiments'''

ax.text(9.9, 10.9, config_text, ha='left', va='top',
        fontsize=6.5, family='monospace', zorder=3)

# Dashed arrows from config to stages
draw_arrow(ax, 9.8, 10.8, 7.1, prep_y, style='--', color=colors['orange'], width=1.2)
draw_arrow(ax, 9.8, 10.6, 7.1, features_y, style='--', color=colors['orange'], width=1.2)
draw_arrow(ax, 9.8, 10.4, 7.1, train_y, style='--', color=colors['orange'], width=1.2)
draw_arrow(ax, 9.8, 10.2, 7.1, threshold_y, style='--', color=colors['orange'], width=1.2)

# Save as PDF
plt.tight_layout()
output_path = '/mnt/c/Users/Jason/predictive-maintenance/docs/conference-templates/data-mining-report/figures/ml_pipeline_flowchart.pdf'
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"✓ Compact pipeline architecture diagram saved to:\n  {output_path}")

plt.close()
