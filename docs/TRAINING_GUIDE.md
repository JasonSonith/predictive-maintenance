# Manual Model Training Guide

This guide shows you how to train each anomaly detection model step-by-step.

---

## Prerequisites

Before training, make sure you have:

1. **Activated your virtual environment**:
```bash
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

2. **Features already extracted**:
```bash
# Check if features exist:
ls data/features/ims/

# Should see:
# - ims_features.parquet
# - ims_features.csv
```

If features don't exist, create them first:
```bash
python scripts/make_features.py --config configs/ims.yaml
```

---

## Training Commands

All training uses the same command format:
```bash
python scripts/train.py --config <path_to_model_config>
```

### 1. Train Isolation Forest (Recommended First)

**Why this model?**
- Fastest to train
- Works well with high-dimensional data
- Good for beginners
- Uses 100 decision trees

**Command:**
```bash
python scripts/train.py --config configs/models/isolation_forest.yaml
```

**What it does:**
- Loads features from `data/features/ims/ims_features.parquet`
- Uses first 20 files (healthy baseline) for training
- Scales features using StandardScaler
- Trains 100 isolation trees
- Saves model to `artifacts/models/ims_iforest.joblib`
- Creates report in `artifacts/reports/ims_iforest/`

**Expected output:**
```
============================================================
ANOMALY DETECTION MODEL TRAINING - STAGE 3
============================================================

Loading model config: configs/models/isolation_forest.yaml
Loading dataset config: configs/ims.yaml
Loading features: data/features/ims/ims_features.parquet
Loaded 14592 feature vectors with 95 columns
Filtering to first 20 files (normal baseline)
Normal baseline: 2400 samples from 20 files
Selected 87 feature columns

Training data shape: (2400, 87)
Features: 87, Samples: 2400

Applying standard scaling...

Training Isolation Forest...
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    2.3s finished
Training complete!

Evaluating on training data...
Training data predictions:
  Normal: 2160 (90.0%)
  Anomaly: 240 (10.0%)
  Score range: [-0.3421, 0.5234]
  Score mean: 0.1234, std: 0.0987

Saving model: artifacts/models/ims_iforest.joblib
Saving scaler: artifacts/models/ims_iforest_scaler.joblib
Saving run metadata: artifacts/reports/ims_iforest/run.json
Saving feature list: artifacts/reports/ims_iforest/features.txt

============================================================
Model training complete!
Model saved: artifacts/models/ims_iforest.joblib
Report dir: artifacts/reports/ims_iforest/
============================================================
```

**Training time:** ~10-30 seconds

---

### 2. Train kNN-LOF (Local Outlier Factor)

**Why this model?**
- Based on k-nearest neighbors
- Good at finding local anomalies
- Considers density of data points
- Slower than Isolation Forest

**Command:**
```bash
python scripts/train.py --config configs/models/knn_lof.yaml
```

**What it does:**
- Same data loading as above
- Uses 20 nearest neighbors to compute local density
- Points in low-density regions = anomalies
- Saves model to `artifacts/models/ims_knn_lof.joblib`

**Training time:** ~30-60 seconds (slower due to distance calculations)

---

### 3. Train One-Class SVM

**Why this model?**
- Creates a boundary around normal data
- Based on support vector machines
- Good for well-separated data
- Slowest of the three

**Command:**
```bash
python scripts/train.py --config configs/models/one_class_svm.yaml
```

**What it does:**
- Same data loading as above
- Uses RBF kernel to create decision boundary
- Points outside boundary = anomalies
- Saves model to `artifacts/models/ims_ocsvm.joblib`

**Training time:** ~1-3 minutes (slowest, especially with large data)

---

## Train All Three Models (Sequential)

To train all models one after another:

```bash
# Train all three models
python scripts/train.py --config configs/models/isolation_forest.yaml && \
python scripts/train.py --config configs/models/knn_lof.yaml && \
python scripts/train.py --config configs/models/one_class_svm.yaml
```

Or create a simple script `train_all.sh`:
```bash
#!/bin/bash
echo "Training Isolation Forest..."
python scripts/train.py --config configs/models/isolation_forest.yaml

echo "Training kNN-LOF..."
python scripts/train.py --config configs/models/knn_lof.yaml

echo "Training One-Class SVM..."
python scripts/train.py --config configs/models/one_class_svm.yaml

echo "All models trained!"
```

Then run:
```bash
chmod +x train_all.sh
./train_all.sh
```

---

## Using the Refactored Version

To use the new class-based version (identical results):

```bash
# Same commands, just use train_refactored.py instead:
python scripts/train_refactored.py --config configs/models/isolation_forest.yaml
python scripts/train_refactored.py --config configs/models/knn_lof.yaml
python scripts/train_refactored.py --config configs/models/one_class_svm.yaml
```

---

## Output Files

After training each model, you'll get:

### 1. Model Files (artifacts/models/)
- `ims_iforest.joblib` - Trained Isolation Forest
- `ims_iforest_scaler.joblib` - Feature scaler
- `ims_knn_lof.joblib` - Trained kNN-LOF
- `ims_knn_lof_scaler.joblib` - Feature scaler
- `ims_ocsvm.joblib` - Trained One-Class SVM
- `ims_ocsvm_scaler.joblib` - Feature scaler

### 2. Report Files (artifacts/reports/<model_name>/)

Each model gets a report directory with:

**run.json** - Training metadata:
```json
{
  "timestamp": "2025-11-21T10:30:45",
  "model_name": "ims_iforest",
  "model_type": "isolation_forest",
  "git_commit": "bf57000abc123",
  "hyperparameters": {
    "n_estimators": 100,
    "contamination": 0.1
  },
  "n_features": 87,
  "n_training_samples": 2400,
  "training_evaluation": {
    "n_normal": 2160,
    "n_anomaly": 240,
    "anomaly_rate": 0.1
  }
}
```

**features.txt** - List of all features used:
```
bearing1_mean
bearing1_std
bearing1_rms
bearing1_peak_to_peak
...
(87 features total)
```

---

## Checking Training Results

After training, verify the outputs:

```bash
# Check model files exist
ls -lh artifacts/models/

# Check report directories
ls artifacts/reports/

# View training metadata for Isolation Forest
cat artifacts/reports/ims_iforest/run.json

# View features used
head -20 artifacts/reports/ims_iforest/features.txt
```

---

## Troubleshooting

### Error: "Features file not found"

**Problem:** Haven't created features yet.

**Solution:**
```bash
python scripts/make_features.py --config configs/ims.yaml
```

---

### Error: "No module named 'sklearn'"

**Problem:** Virtual environment not activated or dependencies not installed.

**Solution:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Error: "Config file not found"

**Problem:** Running from wrong directory.

**Solution:**
```bash
# Make sure you're in the project root
cd /mnt/c/Users/Jason/predictive-maintenance

# Then run training
python scripts/train.py --config configs/models/isolation_forest.yaml
```

---

### Training takes too long (> 5 minutes)

**Problem:** Dataset too large or SVM parameters need tuning.

**Solutions:**
- Start with Isolation Forest (fastest)
- Reduce training samples in config
- Use smaller `max_samples` for Isolation Forest
- For SVM, reduce `max_iter` or use simpler kernel

---

## Customizing Training

### Change Hyperparameters

Edit the model config file:

```bash
# Open config in editor
nano configs/models/isolation_forest.yaml

# Change values:
hyperparameters:
  n_estimators: 200  # More trees (slower but more accurate)
  contamination: 0.05  # Expect 5% anomalies instead of 10%
```

Then retrain:
```bash
python scripts/train.py --config configs/models/isolation_forest.yaml
```

---

### Change Feature Scaling

In the config file:
```yaml
# Options: standard, minmax, robust, none
scaler: minmax  # Scale to 0-1 range instead of z-scores
```

---

### Train on Different Dataset

Change the `dataset_config` in the model config:

```yaml
# Train on CWRU instead of IMS
dataset_config: configs/cwru.yaml
```

Then:
```bash
# Make sure CWRU features exist first
python scripts/make_features.py --config configs/cwru.yaml

# Train
python scripts/train.py --config configs/models/isolation_forest.yaml
```

---

## What Happens During Training?

Here's the step-by-step process:

```
1. LOAD CONFIG
   ↓
   Read model_config.yaml
   Read dataset_config.yaml

2. LOAD FEATURES
   ↓
   Read data/features/ims/ims_features.parquet
   (14,592 windows × 87 features)

3. FILTER TO NORMAL BASELINE
   ↓
   Keep only first 20 files (healthy bearings)
   (2,400 windows × 87 features)

4. SELECT FEATURES
   ↓
   Remove metadata columns (timestamp, file_index)
   Keep only numeric features
   (2,400 windows × 87 features)

5. SCALE FEATURES
   ↓
   Apply StandardScaler
   Mean=0, Std=1 for each feature

6. TRAIN MODEL
   ↓
   Fit Isolation Forest on normal data
   Learn what "healthy" looks like

7. EVALUATE
   ↓
   Predict on training data
   Should flag ~10% as anomalies (contamination param)

8. SAVE
   ↓
   Save model.joblib
   Save scaler.joblib
   Save run.json metadata
   Save features.txt list
```

---

## Next Steps After Training

Once you have trained models:

1. **Set threshold** (Stage 4):
```bash
python scripts/threshold.py --report artifacts/reports/ims_iforest/
```

2. **Evaluate on test data** (Stage 5):
```bash
python scripts/evaluate.py --report artifacts/reports/ims_iforest/
```

3. **Score new data** (Stage 6):
```bash
python scripts/score_batch.py \
  --config configs/ims.yaml \
  --model artifacts/models/ims_iforest.joblib \
  --scaler artifacts/models/ims_iforest_scaler.joblib \
  --input data/features/ims/ims_features.parquet \
  --output artifacts/reports/ims_iforest/scores.csv
```

---

## Quick Reference

| Model | Speed | Accuracy | Best For | Command |
|-------|-------|----------|----------|---------|
| **Isolation Forest** | Fast | Good | High-dimensional, large datasets | `python scripts/train.py --config configs/models/isolation_forest.yaml` |
| **kNN-LOF** | Medium | Better | Local patterns, moderate size | `python scripts/train.py --config configs/models/knn_lof.yaml` |
| **One-Class SVM** | Slow | Best | Well-separated, smaller datasets | `python scripts/train.py --config configs/models/one_class_svm.yaml` |

**Recommended order:**
1. Start with Isolation Forest (fast, reliable)
2. Try kNN-LOF if you have time
3. Try SVM if the other two don't work well

---

## Summary

Training a model is just one command:
```bash
python scripts/train.py --config configs/models/<model_name>.yaml
```

The config file controls everything:
- Which dataset to use
- Which hyperparameters
- Where to save outputs

You don't need to write any code - just run the command and wait!
