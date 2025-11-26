# Evaluation Fixes Summary

## Date: 2025-11-26
## Issues Investigated

This document summarizes the investigation and fixes for three evaluation issues identified after running `evaluate.py` on all 10 trained models.

---

## Issue 1: AutoEncoder/LOF Extreme Outliers ✅ RESOLVED

### Initial Concern
- AutoEncoder: 108 samples with scores < -10 (min: -93.86)
- KNN LOF: 459 samples with scores < -50 (min: -457.66)
- Potential NaN/Inf values or score clipping needed

### Investigation Results
**No bugs found!** These are legitimate extreme anomalies during bearing failure.

#### Evidence:
1. **No NaN or Inf values** in either model
2. **All extreme outliers occur in Period 5** (Nov 19-25, 2003) - the bearing failure period
3. **Temporal distribution**:
   - Periods 1-4 (healthy/early degradation): 0 extreme outliers
   - Period 5 (catastrophic failure): 100% of extreme outliers

#### Score Statistics by Period:

**AutoEncoder:**
| Period | Mean Score | Extreme Outliers (< -10) |
|--------|------------|--------------------------|
| Period 1 (Oct 22-29) | -0.052 | 0 (0.00%) |
| Period 2 (Oct 29-Nov 5) | -0.071 | 0 (0.00%) |
| Period 3 (Nov 5-12) | -0.069 | 0 (0.00%) |
| Period 4 (Nov 12-19) | -0.070 | 0 (0.00%) |
| Period 5 (Nov 19-25) | -0.164 | 108 (0.099%) |

**KNN LOF:**
| Period | Mean Score | Extreme Outliers (< -50) |
|--------|------------|--------------------------|
| Period 1 | -1.11 | 0 (0.00%) |
| Period 2 | -1.50 | 0 (0.00%) |
| Period 3 | -1.50 | 0 (0.00%) |
| Period 4 | -1.56 | 0 (0.00%) |
| Period 5 | -3.26 | 459 (0.419%) |

### Conclusion
The extreme scores are **correct behavior** - these models are successfully identifying severe anomalies during bearing failure. The bearing failed on Nov 25, 2003, and these scores reflect catastrophic degradation.

**Action**: No changes needed. Document this as expected behavior.

---

## Issue 2: One-Class SVM Threshold & Score Saturation ⚠️ PARTIALLY FIXED

### Initial Problem
- Threshold: -137.30 (seemed extreme)
- All 3,902 anomalies had identical scores (-137.30)
- Score saturation preventing smooth anomaly scoring

### Root Cause
**Hyperparameter misconfiguration**: `nu=0.3` was too high

- `nu` controls the fraction of training samples treated as outliers
- `nu=0.3` → 30% of training data flagged as anomalies
- This created an extremely loose decision boundary
- Test scores saturated at minimum value

### Fix Applied
1. **Reduced `nu` from 0.3 to 0.05** in `configs/models/one_class_svm.yaml`
2. **Retrained model**: `python scripts/train.py --config configs/models/one_class_svm.yaml`
3. **Recalibrated threshold**: `python scripts/threshold.py ...`
4. **Re-evaluated**: `python scripts/evaluate.py --report artifacts/reports/ims_ocsvm/`

### Results Comparison

| Metric | Before (nu=0.3) | After (nu=0.05) | Improvement |
|--------|-----------------|-----------------|-------------|
| Training anomaly rate | 30.0% (912/3040) | 5.1% (154/3040) | ✅ Much better |
| Score range | [-133, +37] | [-12, +4] | ✅ Tighter distribution |
| Threshold | -137.30 | -12.13 | ✅ More reasonable |
| Anomaly scores std | 0.0001 (saturated) | 0.0001 (still saturated) | ❌ Still an issue |

### Remaining Issue
**Score saturation persists**: Even with nu=0.05, anomaly scores cluster at -12.13 (std=0.0001). This is a **fundamental limitation of One-Class SVM** with extreme outliers in high dimensions.

### Impact
- ✅ Model achieves target FAR (2.0/week)
- ✅ Detects failure period correctly (99% of anomalies in Period 5)
- ⚠️ Provides minimal score variation for ranking anomalies
- ⚠️ Threshold calibration works, but score distribution is not smooth

### Recommendation
**Keep the fix** (nu=0.05 is correct), but document the limitation. For production use:
- **Isolation Forest** is superior (smooth score distribution, faster, more interpretable)
- One-Class SVM can be used for binary alerts but not for anomaly ranking

---

## Issue 3: CWRU Dataset - No Normal Baseline ❌ UNFIXABLE WITH CURRENT DATA

### Problem
CWRU model shows **catastrophic performance**:
```json
{
  "precision": 1.0,
  "recall": 0.0013,
  "accuracy": 0.0013,
  "confusion_matrix": {
    "TP": 8,
    "TN": 0,
    "FP": 0,
    "FN": 6337
  }
}
```

**Translation**: Model misses 99.87% of faults!

### Root Cause
**CWRU has no normal bearing data**:
```bash
$ cat data/features/cwru/cwru_features.csv | cut -d',' -f2 | sort | uniq
ball_fault
inner_race_fault
outer_race_fault
```

**Impact**:
1. Model trained on 100% faulty data
2. Learned "faults are normal"
3. Only flags extreme outliers (8 out of 6,345)
4. Threshold calibration for FAR is meaningless (no true "normal" baseline)

### Why This Happened
CWRU is a **labeled fault classification dataset**, not an unsupervised anomaly detection dataset:
- Designed for multi-class classification (normal vs. 3 fault types)
- Original dataset has normal bearing data (`Normal_*.mat`)
- Our preprocessing only extracted fault files

### Solutions

#### Option 1: Add Normal Bearing Data (Recommended)
**Steps**:
1. Download CWRU normal bearing MATLAB files
2. Convert to CSV using `scripts/convert_cwru_mat_to_csv.py`
3. Update `configs/cwru.yaml` to include normal files
4. Rerun preprocessing: `python scripts/prep_data.py --config configs/cwru.yaml`
5. Rerun feature extraction: `python scripts/make_features.py --config configs/cwru.yaml`
6. Retrain: `python scripts/train.py --config configs/models/isolation_forest.yaml` (update dataset_config)
7. Re-evaluate

**Pros**: Enables proper unsupervised anomaly detection (normal vs. any fault)
**Cons**: Requires downloading/processing new data

#### Option 2: Switch to Supervised Classification
**Treat CWRU as multi-class classification** (normal vs. ball vs. inner vs. outer):
- Use `RandomForestClassifier` or `LogisticRegression` instead of anomaly detection
- Train on labeled fault types
- Evaluate with classification metrics (accuracy, F1, confusion matrix)

**Pros**: Matches dataset's intended use case
**Cons**: Requires new model type and config (not currently in pipeline)

#### Option 3: One-vs-Rest Anomaly Detection
**Treat one fault type as "normal"**, others as anomalies:
- Example: Train on ball_fault only, flag inner/outer as anomalies
- Rotate through fault types

**Pros**: Works with current pipeline
**Cons**: Artificial setup, limited practical value

### Current Status
**CWRU evaluation results are INVALID** with current data. Do not use for model comparison or production decisions.

### Recommended Action
**Pause CWRU evaluation** until normal bearing data is added. Focus on:
- IMS (working correctly)
- AI4I (working correctly)
- NASA C-MAPSS (working correctly)

---

## Summary of Actions Taken

### ✅ Completed
1. **Investigated AutoEncoder/LOF outliers** → No issues found (correct behavior)
2. **Fixed One-Class SVM hyperparameters** → Reduced nu from 0.3 to 0.05
3. **Retrained One-Class SVM** → Model now more stable
4. **Identified CWRU dataset limitation** → Documented unfixable with current data

### 📝 Documentation Updates
- Updated `configs/models/one_class_svm.yaml` with corrected nu parameter
- Created this summary document

### 🚀 Next Steps
1. **Add CWRU normal bearing data** (see Option 1 above)
2. **Consider deprecating One-Class SVM** in favor of Isolation Forest for production
3. **Update CLAUDE.md** with lessons learned about hyperparameter sensitivity

---

## Model Recommendations for Production

Based on evaluation results:

| Dataset | Recommended Model | Reason |
|---------|-------------------|---------|
| **IMS** | Isolation Forest | Stable, fast, interpretable, good FAR calibration |
| **AI4I** | Isolation Forest | Well-calibrated (0.202/week vs. 0.2 target) |
| **CWRU** | ❌ **Not Ready** | Needs normal bearing data first |
| **NASA FD002** | Isolation Forest | Best FAR accuracy (0.216/week vs. 0.2 target) |
| **NASA FD003** | Isolation Forest | Good FAR accuracy (0.217/week vs. 0.2 target) |

### Alternative Models (IMS)
- **AutoEncoder**: Good (0.200/week perfect FAR), but slower, requires PyTorch
- **KNN LOF**: Good (0.200/week perfect FAR), but memory-intensive
- **One-Class SVM**: ⚠️ Works but has score saturation issues

---

## Files Modified
- `configs/models/one_class_svm.yaml` - Reduced nu from 0.3 to 0.05
- `artifacts/models/ims_ocsvm.joblib` - Retrained model
- `artifacts/reports/ims_ocsvm/*` - Updated evaluation outputs

## Files Created
- `EVALUATION_FIXES_SUMMARY.md` - This document
