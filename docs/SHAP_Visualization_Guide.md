# SHAP Visualization Guide

A step-by-step guide for creating SHAP visualizations and other useful graphs for predictive maintenance presentations.

## Understanding What SHAP Shows

SHAP tells you **which features are most important** for your model's predictions. For example: "Why did the model flag this as an anomaly? Because RMS was very high and kurtosis was unusual."

## Step-by-Step Implementation

### Step 1: Install SHAP (if needed)

```bash
pip install shap matplotlib
```

### Step 2: Create a Simple Script

Create a new file called `visualize_shap.py` in your `scripts/` folder:

```python
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. Load your trained model
model = joblib.load('artifacts/models/ims_iforest.joblib')

# 2. Load your test data (the features)
X_test = pd.read_csv('data/features/ims_test.csv')

# 3. Remove non-feature columns (like timestamps)
feature_cols = [col for col in X_test.columns
                if col not in ['timestamp', 'file_id', 'unit_id', 'anomaly']]
X_features = X_test[feature_cols]

# Take a small sample (SHAP is slow on large data)
X_sample = X_features.head(100)

# 4. Create SHAP explainer
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

# 5. Create visualizations

# --- Plot 1: Summary Plot (shows top features) ---
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig('artifacts/figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Plot 2: Bar Plot (simpler version) ---
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('artifacts/figures/shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ SHAP plots saved to artifacts/figures/")
```

### Step 3: Run It

```bash
python scripts/visualize_shap.py
```

You'll get two images:
- **shap_summary.png**: Shows which features matter most (with colors showing high/low values)
- **shap_bar.png**: Simple bar chart of feature importance

---

## Other Useful Visualizations

Here are more graphs you can create for your presentation:

### Anomaly Score Over Time

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load scored data
scores = pd.read_csv('artifacts/scores/ims_test_scores.csv')

plt.figure(figsize=(12, 5))
plt.plot(scores['timestamp'], scores['anomaly_score'], linewidth=1)
plt.axhline(y=0.485, color='r', linestyle='--', label='Threshold')
plt.xlabel('Time')
plt.ylabel('Anomaly Score')
plt.title('Anomaly Detection Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('artifacts/figures/anomaly_timeline.png', dpi=300)
print("✅ Saved anomaly timeline")
```

### Confusion Matrix (if you have labels)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming you have true labels
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('artifacts/figures/confusion_matrix.png', dpi=300)
```

### Feature Distribution Comparison

```python
import seaborn as sns

# Compare normal vs anomaly feature values
normal_data = X_test[y_pred == 0]
anomaly_data = X_test[y_pred == 1]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features_to_plot = ['rms', 'kurtosis', 'peak_to_peak', 'std']

for ax, feature in zip(axes.flat, features_to_plot):
    ax.hist(normal_data[feature], bins=50, alpha=0.5, label='Normal')
    ax.hist(anomaly_data[feature], bins=50, alpha=0.5, label='Anomaly')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.savefig('artifacts/figures/feature_distributions.png', dpi=300)
```

---

## Quick Reference: What Graph to Use When

| Want to show... | Use this... |
|----------------|-------------|
| Which features matter most | SHAP bar plot |
| How features interact | SHAP summary plot (beeswarm) |
| Anomalies over time | Line plot with threshold |
| Model performance | Confusion matrix, ROC curve |
| False alarm rate | Histogram of scores by class |
| Feature correlations | Heatmap (seaborn) |

---

## Pro Tips

1. **Start with SHAP bar plot** - it's the easiest to explain
2. **Use high DPI** (300) for presentations
3. **Keep it simple** - 3-4 key plots beat 10 confusing ones
4. **Add clear titles and labels** - your audience shouldn't guess
5. **Use consistent color schemes** across all plots
6. **Export as PNG** for PowerPoint (better than PDF for slides)
7. **Test on sample data first** - SHAP can be slow on 10,000+ samples

---

## Common Issues and Solutions

### Issue: SHAP is too slow
**Solution**: Use a smaller sample (100-500 rows) or use `shap.sample()`:
```python
X_sample = shap.sample(X_test, 200)
```

### Issue: "Explainer not compatible with this model"
**Solution**: Use `shap.Explainer()` (auto-detects) or specify:
- `shap.TreeExplainer()` for tree models (IForest, RandomForest)
- `shap.KernelExplainer()` for any model (slower but universal)

### Issue: Features not aligned
**Solution**: Ensure feature columns match exactly between training and test data:
```python
# Load feature names from model training
with open('artifacts/reports/ims_iforest/feature_names.json') as f:
    feature_names = json.load(f)
X_test = X_test[feature_names]
```

### Issue: Plot labels are cut off
**Solution**: Use `bbox_inches='tight'` in `savefig()`:
```python
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

---

## Example: Complete Visualization Script

Here's a complete script that generates all key plots:

```python
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Setup
OUTPUT_DIR = Path('artifacts/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model and data
print("Loading model and data...")
model = joblib.load('artifacts/models/ims_iforest.joblib')
X_test = pd.read_csv('data/features/ims_test.csv')
scores = pd.read_csv('artifacts/scores/ims_test_scores.csv')

# Extract features
feature_cols = [col for col in X_test.columns
                if col not in ['timestamp', 'file_id', 'unit_id', 'anomaly']]
X_features = X_test[feature_cols]

# 1. SHAP Summary (sample for speed)
print("Generating SHAP plots...")
X_sample = X_features.head(200)
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved SHAP bar plot")

# 2. Anomaly Score Timeline
print("Generating timeline plot...")
plt.figure(figsize=(12, 5))
plt.plot(range(len(scores)), scores['anomaly_score'], linewidth=1, alpha=0.7)
plt.axhline(y=0.485, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score')
plt.title('Anomaly Score Over Time - IMS Dataset')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'anomaly_timeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved timeline plot")

# 3. Score Distribution
print("Generating score distribution...")
plt.figure(figsize=(10, 6))
plt.hist(scores['anomaly_score'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0.485, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Distribution of Anomaly Scores')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'score_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved score distribution")

# 4. Top 10 Features Bar Chart
print("Generating feature importance...")
feature_importance = np.abs(shap_values.values).mean(axis=0)
feature_names = X_sample.columns
top_indices = np.argsort(feature_importance)[-10:]

plt.figure(figsize=(10, 6))
plt.barh(range(10), feature_importance[top_indices])
plt.yticks(range(10), feature_names[top_indices])
plt.xlabel('Mean |SHAP Value|')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved top features plot")

print(f"\n✅ All plots saved to {OUTPUT_DIR}/")
```

Run with:
```bash
python scripts/visualize_all.py
```

---

## For Your Presentation

### Recommended Slides

1. **Feature Importance Slide**
   - Use: `shap_bar.png`
   - Message: "RMS and kurtosis are the top predictors"

2. **Detection Performance Slide**
   - Use: `anomaly_timeline.png`
   - Message: "Model successfully detects degradation before failure"

3. **Score Distribution Slide**
   - Use: `score_distribution.png`
   - Message: "Clear separation between normal and anomaly scores"

4. **Methodology Slide**
   - Use: Pipeline diagram (from existing docs)
   - Message: "6-stage pipeline from raw data to production"

### Color Scheme Suggestions

```python
# Professional color palette
COLORS = {
    'primary': '#2E86AB',    # Blue
    'danger': '#A23B72',     # Red/Purple
    'success': '#06A77D',    # Green
    'warning': '#F18F01',    # Orange
    'neutral': '#6C757D'     # Gray
}

# Use in plots
plt.plot(..., color=COLORS['primary'])
plt.axhline(..., color=COLORS['danger'])
```

---

## Additional Resources

- **SHAP Documentation**: https://shap.readthedocs.io/
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
- **Seaborn Tutorial**: https://seaborn.pydata.org/tutorial.html
- **Your Pipeline Docs**: `docs/Project-breakdown.md`
