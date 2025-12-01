#!/usr/bin/env python3
"""
Generate polished, professional predictive maintenance pipeline diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.patheffects import withStroke

# Professional figure size with golden ratio proportions
fig, ax = plt.subplots(figsize=(7, 9.5), facecolor='white')
ax.set_xlim(0, 7)
ax.set_ylim(0, 9.5)
ax.axis('off')

# Professional color palette - muted, sophisticated
colors = {
    'datasets': '#3B82F6',      # Professional blue
    'stage': '#10B981',          # Professional green
    'stage_dark': '#059669',     # Darker green
    'model': '#F59E0B',          # Amber
    'config': '#8B5CF6',         # Purple
    'production': '#EC4899',     # Pink
    'text': '#1F2937',           # Dark gray
    'arrow': '#6B7280',          # Medium gray
    'accent': '#EF4444'          # Red accent
}

def draw_stage_box(ax, x, y, width, height, title, subtitle='', color=colors['stage']):
    """Draw polished stage box with subtle shadow"""
    # Shadow
    shadow = FancyBboxPatch((x - width/2 + 0.04, y - height/2 - 0.04), width, height,
                           boxstyle='round,pad=0.02',
                           facecolor='black', alpha=0.08, zorder=1,
                           linewidth=0)
    ax.add_patch(shadow)

    # Main box with gradient effect (simulated with alpha)
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle='round,pad=0.02',
                         facecolor=color, edgecolor='white',
                         linewidth=2.5, alpha=0.95, zorder=2)
    ax.add_patch(box)

    # Title with text stroke for better readability
    text_offset = 0.08 if subtitle else 0
    title_obj = ax.text(x, y + text_offset, title, ha='center', va='center',
                       fontsize=9.5, color='white', fontweight='600', zorder=3,
                       family='sans-serif')
    title_obj.set_path_effects([withStroke(linewidth=3, foreground='black', alpha=0.3)])

    # Subtitle
    if subtitle:
        sub_obj = ax.text(x, y - 0.08, subtitle, ha='center', va='center',
                         fontsize=7.5, color='white', alpha=0.9, zorder=3,
                         family='monospace', style='italic')
        sub_obj.set_path_effects([withStroke(linewidth=2, foreground='black', alpha=0.2)])

def draw_data_box(ax, x, y, width, height, text, color=colors['datasets']):
    """Draw compact data box"""
    # Shadow
    shadow = Rectangle((x - width/2 + 0.03, y - height/2 - 0.03), width, height,
                      facecolor='black', alpha=0.06, zorder=1)
    ax.add_patch(shadow)

    # Main box
    box = Rectangle((x - width/2, y - height/2), width, height,
                   facecolor=color, edgecolor='white',
                   linewidth=2, alpha=0.9, zorder=2)
    ax.add_patch(box)

    # Text
    text_obj = ax.text(x, y, text, ha='center', va='center',
                      fontsize=7.5, color='white', fontweight='600', zorder=3)
    text_obj.set_path_effects([withStroke(linewidth=2, foreground='black', alpha=0.25)])

def draw_arrow(ax, x1, y1, x2, y2, color=colors['arrow'], width=2.5, style='-', alpha=1.0):
    """Draw elegant arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=width, color=color, alpha=alpha,
                           linestyle=style, zorder=1,
                           connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)

# Layout positions
center_x = 3.5
stage_width = 5.5
stage_height = 0.55
gap = 0.85

y_pos = 9.0
datasets_y = y_pos; y_pos -= gap
prep_y = y_pos; y_pos -= gap
features_y = y_pos; y_pos -= gap
train_y = y_pos; y_pos -= gap + 0.2
threshold_y = y_pos; y_pos -= gap
evaluate_y = y_pos; y_pos -= gap
production_y = y_pos - 0.3

# ===== DATASETS =====
datasets_text = "Input Datasets\nIMS · CWRU · AI4I · C-MAPSS"
draw_data_box(ax, center_x, datasets_y, 5, 0.5, datasets_text, colors['datasets'])
draw_arrow(ax, center_x, datasets_y - 0.28, center_x, prep_y + 0.3)

# ===== STAGE 1: Data Preparation =====
draw_stage_box(ax, center_x, prep_y, stage_width, stage_height,
               "1. Data Preparation", "prep_data.py")
draw_arrow(ax, center_x, prep_y - 0.3, center_x, features_y + 0.3)

# ===== STAGE 2: Feature Engineering =====
draw_stage_box(ax, center_x, features_y, stage_width, stage_height,
               "2. Feature Engineering", "2048-sample windows, 50% overlap")
draw_arrow(ax, center_x, features_y - 0.3, center_x, train_y + 0.38)

# ===== STAGE 3: Model Training =====
# Main training box
draw_stage_box(ax, center_x, train_y, stage_width, 0.7,
               "3. Model Training", "train.py")

# Four models in compact layout
model_names = ["IForest", "LOF", "SVM", "AE"]
model_x = [2.3, 3.2, 4.1, 5.0]
for name, mx in zip(model_names, model_x):
    model_box = Rectangle((mx - 0.35, train_y - 0.48), 0.7, 0.28,
                          facecolor=colors['model'], edgecolor='white',
                          linewidth=1.5, alpha=0.85, zorder=3)
    ax.add_patch(model_box)
    ax.text(mx, train_y - 0.34, name, ha='center', va='center',
            fontsize=6.5, color='white', fontweight='600', zorder=4)

draw_arrow(ax, center_x, train_y - 0.38, center_x, threshold_y + 0.3)

# ===== STAGE 4: Threshold Calibration =====
draw_stage_box(ax, center_x, threshold_y, stage_width, stage_height,
               "4. Threshold Calibration", "FAR: 0.2 alarms/week")
draw_arrow(ax, center_x, threshold_y - 0.3, center_x, evaluate_y + 0.3)

# ===== STAGE 5: Evaluation =====
draw_stage_box(ax, center_x, evaluate_y, stage_width, stage_height,
               "5. Evaluation & Reporting", "Metrics · SHAP · Visualizations")
draw_arrow(ax, center_x, evaluate_y - 0.3, center_x, production_y + 0.3)

# ===== STAGE 6: Production =====
# Dashed production box
prod_shadow = FancyBboxPatch((center_x - stage_width/2 + 0.04, production_y - 0.28 - 0.04),
                            stage_width, 0.55,
                            boxstyle='round,pad=0.02',
                            facecolor='black', alpha=0.08, zorder=1, linewidth=0)
ax.add_patch(prod_shadow)

prod_box = FancyBboxPatch((center_x - stage_width/2, production_y - 0.28),
                         stage_width, 0.55,
                         boxstyle='round,pad=0.02',
                         facecolor=colors['production'], edgecolor='white',
                         linewidth=2.5, linestyle='--', alpha=0.95, zorder=2)
ax.add_patch(prod_box)

prod_text = ax.text(center_x, production_y + 0.08, "6. Production Scoring",
                   ha='center', va='center', fontsize=9.5, color='white',
                   fontweight='600', zorder=3)
prod_text.set_path_effects([withStroke(linewidth=3, foreground='black', alpha=0.3)])

prod_sub = ax.text(center_x, production_y - 0.08, "score_batch.py · Real-time alerts",
                  ha='center', va='center', fontsize=7.5, color='white',
                  alpha=0.9, zorder=3, family='monospace', style='italic')
prod_sub.set_path_effects([withStroke(linewidth=2, foreground='black', alpha=0.2)])

# Input/Output labels
io_box_props = dict(boxstyle='round,pad=0.08', facecolor=colors['datasets'],
                    alpha=0.85, edgecolor='white', linewidth=1.8)

ax.text(0.85, production_y, "New\nData", ha='center', va='center',
        fontsize=7.5, color='white', fontweight='600', bbox=io_box_props)

ax.text(6.15, production_y, "Alerts", ha='center', va='center',
        fontsize=7.5, color='white', fontweight='600', bbox=io_box_props)

draw_arrow(ax, 1.35, production_y, center_x - stage_width/2 - 0.05, production_y, width=2)
draw_arrow(ax, center_x + stage_width/2 + 0.05, production_y, 5.75, production_y, width=2)

# ===== CONFIG PANEL (elegant sidebar) =====
config_panel = FancyBboxPatch((6.0, 5.0), 0.85, 3.5,
                              boxstyle='round,pad=0.05',
                              facecolor=colors['config'], edgecolor='white',
                              linewidth=2, alpha=0.15, zorder=0)
ax.add_patch(config_panel)

# Config title
config_title = ax.text(6.425, 8.2, "Config", ha='center', va='center',
                      fontsize=8, color=colors['config'], fontweight='700', zorder=1)

# Config items (minimalist)
config_items = [
    ("YAML", 7.8),
    ("Dataset", 7.3),
    ("Model", 6.8),
    ("Params", 6.3)
]

for item, y in config_items:
    ax.text(6.425, y, item, ha='center', va='center',
            fontsize=6.5, color=colors['config'], alpha=0.7, zorder=1)

# Elegant connection lines from config to stages
config_stages = [prep_y, features_y, train_y, threshold_y]
for stage_y in config_stages:
    draw_arrow(ax, 6.0, 6.8, 5.82, stage_y,
              color=colors['config'], width=1.2, style=':', alpha=0.4)

# Title annotation (optional - can be removed if too much)
title_text = ax.text(3.5, 9.65, "Predictive Maintenance Pipeline",
                    ha='center', va='center', fontsize=11,
                    color=colors['text'], fontweight='700',
                    family='sans-serif')

# Save with high quality
output_path = '/mnt/c/Users/Jason/predictive-maintenance/docs/conference-templates/data-mining-report/figures/ml_pipeline_flowchart.pdf'
plt.tight_layout(pad=0.3)
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.1)
print(f"✓ Polished pipeline diagram saved to:\n  {output_path}")
plt.close()
