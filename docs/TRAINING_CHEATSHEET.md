# Training Cheatsheet - Quick Reference

## TL;DR - Train All Three Models

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the automated script
./train_commands.sh

# Done! All three models trained.
```

---

## Manual Commands (One at a Time)

### Option A: Train Just One Model

```bash
# Isolation Forest (FASTEST - Recommended first)
python scripts/train.py --config configs/models/isolation_forest.yaml

# kNN-LOF (Medium speed)
python scripts/train.py --config configs/models/knn_lof.yaml

# One-Class SVM (Slowest)
python scripts/train.py --config configs/models/one_class_svm.yaml
```

### Option B: Train All Three (Sequential)

```bash
python scripts/train.py --config configs/models/isolation_forest.yaml && \
python scripts/train.py --config configs/models/knn_lof.yaml && \
python scripts/train.py --config configs/models/one_class_svm.yaml
```

---

## What You Get After Training

```
artifacts/
├── models/
│   ├── ims_iforest.joblib          ← Trained model
│   ├── ims_iforest_scaler.joblib   ← Feature scaler
│   ├── ims_knn_lof.joblib
│   ├── ims_knn_lof_scaler.joblib
│   ├── ims_ocsvm.joblib
│   └── ims_ocsvm_scaler.joblib
└── reports/
    ├── ims_iforest/
    │   ├── run.json       ← Training metadata
    │   └── features.txt   ← Features used
    ├── ims_knn_lof/
    │   ├── run.json
    │   └── features.txt
    └── ims_ocsvm/
        ├── run.json
        └── features.txt
```

---

## Model Comparison

| Model | Command | Time | Best For |
|-------|---------|------|----------|
| **Isolation Forest** | `python scripts/train.py --config configs/models/isolation_forest.yaml` | 10-30s | Most datasets (start here!) |
| **kNN-LOF** | `python scripts/train.py --config configs/models/knn_lof.yaml` | 30-60s | Local patterns |
| **One-Class SVM** | `python scripts/train.py --config configs/models/one_class_svm.yaml` | 1-3m | Well-separated data |

---

## Check Training Results

```bash
# List trained models
ls -lh artifacts/models/

# View training metadata
cat artifacts/reports/ims_iforest/run.json

# View features used
head artifacts/reports/ims_iforest/features.txt

# See all reports
ls artifacts/reports/
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| "Features file not found" | Run: `python scripts/make_features.py --config configs/ims.yaml` |
| "No module named sklearn" | Run: `pip install -r requirements.txt` |
| Virtual env not activated | Run: `source .venv/bin/activate` |
| Wrong directory | `cd /mnt/c/Users/Jason/predictive-maintenance` |

---

## Before Training Checklist

- [ ] Virtual environment activated (`source .venv/bin/activate`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Features extracted (`ls data/features/ims/ims_features.parquet`)
- [ ] In project root directory (`pwd` shows `/mnt/c/Users/Jason/predictive-maintenance`)

---

## Training Process Flow

```
1. Load config file
   ↓
2. Load features (data/features/ims/ims_features.parquet)
   ↓
3. Filter to normal baseline (first 20 files)
   ↓
4. Select feature columns (remove metadata)
   ↓
5. Scale features (StandardScaler)
   ↓
6. Train model (fit on normal data)
   ↓
7. Evaluate on training data
   ↓
8. Save model + scaler + metadata
```

---

## Expected Console Output (Isolation Forest)

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

Saving model: artifacts/models/ims_iforest.joblib
Saving scaler: artifacts/models/ims_iforest_scaler.joblib

============================================================
Model training complete!
============================================================
```

---

## Next Steps After Training

```bash
# Stage 4: Set threshold
python scripts/threshold.py --report artifacts/reports/ims_iforest/

# Stage 5: Evaluate
python scripts/evaluate.py --report artifacts/reports/ims_iforest/

# Stage 6: Score new data
python scripts/score_batch.py \
  --model artifacts/models/ims_iforest.joblib \
  --input data/features/ims/ims_features.parquet \
  --output artifacts/reports/ims_iforest/scores.csv
```

---

## Keyboard Shortcuts

**Windows (WSL):**
- Stop training: `Ctrl + C`
- Copy from terminal: `Ctrl + Shift + C`
- Paste to terminal: `Ctrl + Shift + V`

**Terminal Navigation:**
- Up arrow: Previous command
- Tab: Autocomplete paths/filenames
- `Ctrl + R`: Search command history

---

## Pro Tips

1. **Start with Isolation Forest** - It's the fastest and most reliable
2. **Check features first** - `ls data/features/ims/` before training
3. **Use the script** - `./train_commands.sh` trains all three automatically
4. **Monitor progress** - Watch the console output for errors
5. **Compare models later** - Train all three, then evaluate which performs best

---

## Need More Details?

See `TRAINING_GUIDE.md` for:
- Detailed explanations
- Customizing hyperparameters
- Troubleshooting guide
- Training on different datasets
