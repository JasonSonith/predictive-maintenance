# AutoEncoder Training Guide

## What Was Added

The `AutoEncoderTrainer` class has been added to `scripts/train.py` (lines 380-740). Now you have **4 models** available:

1. ✅ Isolation Forest (sklearn)
2. ✅ kNN-LOF (sklearn)
3. ✅ One-Class SVM (sklearn)
4. ✅ **AutoEncoder (PyTorch)** ← NEW!

---

## How AutoEncoder Works (Simple Explanation)

### The Concept

Imagine you're learning to draw portraits:
1. **Encoder**: Look at a face and remember just the key features (eyes, nose, mouth position)
2. **Bottleneck**: Force yourself to remember only 8 numbers (compressed memory)
3. **Decoder**: Try to redraw the face from just those 8 numbers

**Normal faces**: Easy to compress and redraw (low error)
**Weird/broken faces**: Hard to compress, drawing looks bad (high error = anomaly!)

### The Architecture

```
Input (87 features)
    ↓
Encoder Layer 1 (64 neurons) ──┐
    ↓                          │
Encoder Layer 2 (32 neurons)   │ Learns to compress
    ↓                          │
Encoder Layer 3 (16 neurons) ──┘
    ↓
Bottleneck (8 neurons) ← Compressed representation
    ↓
Decoder Layer 1 (16 neurons) ──┐
    ↓                          │
Decoder Layer 2 (32 neurons)   │ Learns to reconstruct
    ↓                          │
Decoder Layer 3 (64 neurons) ──┘
    ↓
Output (87 features)

Reconstruction Error = |Input - Output|
High error = Anomaly!
```

---

## Installation

AutoEncoder requires PyTorch. Install it:

```bash
# Activate your environment
source .venv/bin/activate

# Install PyTorch (CPU version)
pip install torch torchvision

# Or uncomment torch in requirements.txt and run:
# pip install -r requirements.txt
```

---

## Training the AutoEncoder

Same command as other models:

```bash
python scripts/train.py --config configs/models/autoencoder.yaml
```

### What Happens During Training

```
1. Load and prepare data (same as other models)
   ↓
2. Split into training (90%) and validation (10%)
   ↓
3. Create neural network (encoder + decoder)
   ↓
4. Training loop (up to 50 epochs):
   - Forward pass: Input → Encoder → Bottleneck → Decoder → Output
   - Calculate loss: MSE(Input, Output)
   - Backward pass: Update network weights
   - Check validation loss
   ↓
5. Early stopping (stops if no improvement for 10 epochs)
   ↓
6. Set anomaly threshold (at 90th percentile of reconstruction errors)
   ↓
7. Evaluate and save model (.pth file)
```

---

## Configuration (configs/models/autoencoder.yaml)

Key hyperparameters:

```yaml
hyperparameters:
  # Network size
  encoder_dims: [64, 32, 16]  # Layers in encoder
  bottleneck_dim: 8            # Compressed representation size

  # Training settings
  epochs: 50                   # Max training epochs
  batch_size: 32              # Samples per batch
  learning_rate: 0.001        # How fast to learn

  # Regularization
  dropout: 0.2                # Prevents overfitting (0.0-0.5)
  weight_decay: 0.0001        # L2 regularization

  # Early stopping
  early_stopping_patience: 10  # Stop if no improvement
  validation_split: 0.1        # 10% for validation

  # Anomaly detection
  contamination: 0.1           # Expected % of anomalies
```

---

## Output Files

After training:

```
artifacts/models/
  └── ims_autoencoder.pth        ← PyTorch model (not .joblib!)
  └── ims_autoencoder_scaler.joblib

artifacts/reports/ims_autoencoder/
  ├── run.json                   ← Includes training history
  └── features.txt
```

**Note**: AutoEncoder saves as `.pth` (PyTorch format), not `.joblib` like sklearn models.

---

## Expected Training Output

```
============================================================
ANOMALY DETECTION MODEL TRAINING - STAGE 3
============================================================

Loading model config: configs/models/autoencoder.yaml
Loading dataset config: configs/ims.yaml
Loading features: data/features/ims/ims_features.parquet
Loaded 14592 feature vectors with 95 columns

Filtering to first 20 files (normal baseline)
Normal baseline: 2400 samples from 20 files

Selected 87 feature columns

Training data shape: (2400, 87)
Features: 87, Samples: 2400

Applying minmax scaling...

Training samples: 2160, Validation samples: 240
Using device: cpu

Training autoencoder...
Model architecture:
Sequential(
  (0): Linear(in_features=87, out_features=64, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.2, inplace=False)
  (3): Linear(in_features=64, out_features=32, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.2, inplace=False)
  (6): Linear(in_features=32, out_features=16, bias=True)
  (7): ReLU()
  (8): Dropout(p=0.2, inplace=False)
  (9): Linear(in_features=16, out_features=8, bias=True)
  (10): ReLU()
  (11): Linear(in_features=8, out_features=16, bias=True)
  (12): ReLU()
  (13): Dropout(p=0.2, inplace=False)
  (14): Linear(in_features=16, out_features=32, bias=True)
  (15): ReLU()
  (16): Dropout(p=0.2, inplace=False)
  (17): Linear(in_features=32, out_features=64, bias=True)
  (18): ReLU()
  (19): Dropout(p=0.2, inplace=False)
  (20): Linear(in_features=64, out_features=87, bias=True)
)

Training for up to 50 epochs (early stopping patience: 10)...
Epoch 1/50 - Train Loss: 0.045632, Val Loss: 0.038941
Epoch 5/50 - Train Loss: 0.023456, Val Loss: 0.021234
Epoch 10/50 - Train Loss: 0.015678, Val Loss: 0.014523
Epoch 15/50 - Train Loss: 0.012345, Val Loss: 0.011234
Epoch 20/50 - Train Loss: 0.010987, Val Loss: 0.010456
Early stopping at epoch 25 (no improvement for 10 epochs)

Training complete! Best validation loss: 0.010123

Anomaly threshold set at: 0.023456 (contamination: 0.1)

Evaluating on training data...
Training data predictions:
  Normal: 2160 (90.0%)
  Anomaly: 240 (10.0%)
  Reconstruction error range: [0.004523, 0.098765]
  Reconstruction error mean: 0.012345, std: 0.008765
  Threshold: 0.023456

Saving model: artifacts/models/ims_autoencoder.pth
Saving scaler: artifacts/models/ims_autoencoder_scaler.joblib
Saving run metadata: artifacts/reports/ims_autoencoder/run.json
Saving feature list: artifacts/reports/ims_autoencoder/features.txt

============================================================
Model training complete!
Model saved: artifacts/models/ims_autoencoder.pth
Report dir: artifacts/reports/ims_autoencoder/
============================================================
```

---

## Training Time

**Expected time**: 2-5 minutes (depends on epochs and early stopping)

Slower than sklearn models because:
- Neural network training loop
- Multiple epochs (up to 50)
- Batch processing
- Backpropagation calculations

But usually stops early (10-25 epochs) due to early stopping.

---

## Why Use AutoEncoder vs Others?

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **Isolation Forest** | Fast, simple | Less flexible | Quick baseline |
| **kNN-LOF** | Good for local patterns | Slow with large data | Density-based anomalies |
| **One-Class SVM** | Creates clear boundary | Slowest, hard to tune | Well-separated data |
| **AutoEncoder** | Learns complex patterns, flexible | Needs tuning, slower | Complex non-linear patterns |

**Use AutoEncoder when**:
- Other models don't work well
- You have complex, high-dimensional data
- You want to learn non-linear patterns
- You have time to tune hyperparameters

---

## Troubleshooting

### Error: "PyTorch is required for AutoEncoder"

**Solution**:
```bash
pip install torch torchvision
```

---

### Training is too slow (> 10 minutes)

**Solution**: Reduce epochs or increase early stopping patience
```yaml
hyperparameters:
  epochs: 20                      # Down from 50
  early_stopping_patience: 5      # Down from 10
```

---

### Model not learning (loss not decreasing)

**Solution**: Adjust learning rate or architecture
```yaml
hyperparameters:
  learning_rate: 0.01             # Increase from 0.001
  encoder_dims: [128, 64, 32]     # Larger network
```

---

### Too many anomalies detected (> 20%)

**Solution**: Adjust contamination threshold
```yaml
hyperparameters:
  contamination: 0.05  # Down from 0.1 (stricter)
```

---

## Key Differences from Sklearn Models

### 1. Custom Training Loop
- Sklearn: `model.fit(X)` → done
- AutoEncoder: Epochs, batches, early stopping

### 2. Save Format
- Sklearn: `.joblib` (pickle format)
- AutoEncoder: `.pth` (PyTorch state dict)

### 3. Evaluation Method
- Sklearn: Built-in `predict()` and `score_samples()`
- AutoEncoder: Calculate reconstruction error manually

### 4. Threshold Setting
- Sklearn: Built into model (contamination param)
- AutoEncoder: Calculated after training (percentile of errors)

---

## The Hash/Dictionary (MODEL_TRAINERS)

Now includes AutoEncoder:

```python
MODEL_TRAINERS = {
    'isolation_forest': IsolationForestTrainer,
    'knn_lof': KNNLOFTrainer,
    'one_class_svm': OneClassSVMTrainer,
    'autoencoder': AutoEncoderTrainer  # ← NEW!
}
```

When you run:
```bash
python scripts/train.py --config configs/models/autoencoder.yaml
```

1. Reads config: `model_type = 'autoencoder'`
2. Looks up hash: `MODEL_TRAINERS['autoencoder']` → `AutoEncoderTrainer`
3. Creates trainer: `trainer = AutoEncoderTrainer(config)`
4. Trains: `trainer.train()`

Same pattern as before, just added one more entry to the hash!

---

## Complete Training Commands

Train all 4 models:

```bash
# 1. Isolation Forest (30 seconds)
python scripts/train.py --config configs/models/isolation_forest.yaml

# 2. kNN-LOF (1 minute)
python scripts/train.py --config configs/models/knn_lof.yaml

# 3. One-Class SVM (2 minutes)
python scripts/train.py --config configs/models/one_class_svm.yaml

# 4. AutoEncoder (3 minutes)
python scripts/train.py --config configs/models/autoencoder.yaml
```

---

## Summary

✅ **What**: Added neural network-based anomaly detection using reconstruction error

✅ **Where**: Added `AutoEncoderTrainer` class to `scripts/train.py` (lines 380-740)

✅ **How**: Same command as other models, just uses PyTorch instead of sklearn

✅ **Why**: Learns complex non-linear patterns that other models might miss

✅ **Hash**: Added `'autoencoder': AutoEncoderTrainer` to `MODEL_TRAINERS` dictionary

Now you have 4 complete anomaly detection models to choose from! 🎉
