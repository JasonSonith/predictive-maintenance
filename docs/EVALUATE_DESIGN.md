# evaluate.py System Design

## Overview

Stage 5 of the predictive maintenance pipeline. Evaluates trained models on test data and generates comprehensive reports including metrics, visualizations, and SHAP explanations.

**Purpose**: Transform trained models into actionable insights by quantifying performance, visualizing behavior, and explaining predictions.

## Command-Line Interface

```bash
# Simple usage - auto-detect all files from model directory
python scripts/evaluate.py --model_dir artifacts/models/ims_iforest/

# Advanced usage - specify components individually
python scripts/evaluate.py \
  --model artifacts/models/ims_iforest/model.joblib \
  --threshold artifacts/models/ims_iforest/threshold.json \
  --config configs/ims.yaml \
  --output artifacts/reports/ims_iforest/
```

### Key Arguments

- `--model_dir`: Directory containing all model artifacts (simplest)
- `--splits`: Evaluate on train/val/test (default: test only)
- `--shap_samples`: Number of samples for SHAP analysis (default: 100)
- `--save_predictions`: Export full predictions CSV

## Inputs & Outputs

### Inputs
1. **Trained Model** (.joblib file)
2. **Threshold Configuration** (.json with calibrated threshold + FAR target)
3. **Dataset Configuration** (original YAML)
4. **Feature Data** (Parquet files with train/val/test splits)

### Outputs (artifacts/reports/<model_name>/)
- **metrics.json**: All computed metrics in structured format
- **predictions.csv**: Full prediction results with scores and labels
- **7 Visualization Types**: Distribution, timeline, confusion matrix, ROC, PR curve, threshold sensitivity, feature correlation
- **SHAP Explanations**: Summary plot + individual waterfall plots + values CSV
- **report.html**: Comprehensive HTML report with all above content
- **run.json**: Reproducibility metadata (git commit, config, timestamp)

## Architecture

### Three-Layer Design

1. **Evaluation Orchestrator** (`ModelEvaluator`)
   - Coordinates the entire evaluation pipeline
   - Loads models, thresholds, configs, and data
   - Executes evaluation flow: predict → metrics → visualize → explain → report

2. **Metric Computation Layer**
   - Detects dataset type (supervised vs unsupervised, time-series vs tabular)
   - Computes appropriate metrics based on data characteristics
   - Handles missing labels gracefully (degrade to unsupervised metrics)

3. **Output Generation Layer**
   - Creates publication-quality visualizations
   - Generates SHAP explanations (global + individual)
   - Assembles HTML reports with embedded content

### Evaluation Flow

```
Load Model & Config → Load Test Data → Generate Predictions
    ↓
Compute Metrics (adapt to dataset type)
    ↓
Create Visualizations (7 plot types)
    ↓
Generate SHAP Explanations (feature importance + individual cases)
    ↓
Assemble HTML Report → Save All Artifacts
```

## Metrics Strategy

### Dataset-Adaptive Metrics

The system automatically selects appropriate metrics based on dataset characteristics:

**Unsupervised (IMS):**
- Anomaly rate and score statistics
- Temporal progression patterns
- Alert clustering (duration, frequency)
- Estimated false-alarm rate from threshold calibration

**Supervised Classification (CWRU, AI4I):**
- Confusion matrix (TP, TN, FP, FN)
- Precision, Recall, F1, Accuracy
- ROC-AUC and PR-AUC
- Per-class metrics for multi-class scenarios

**Regression → Anomaly (C-MAPSS):**
- MAE, RMSE, R² for RUL prediction
- Correlation between anomaly scores and RUL
- NASA asymmetric scoring function

## Visualization Suite

Seven complementary visualizations provide complete model understanding:

1. **Score Distribution**: Histogram showing anomaly score spread with threshold marker
2. **Timeline Plot**: Temporal evolution of scores (critical for time-series)
3. **Confusion Matrix**: Classification performance breakdown (supervised only)
4. **ROC Curve**: TPR vs FPR with AUC score
5. **Precision-Recall Curve**: Better for imbalanced datasets
6. **Threshold Sensitivity**: How metrics change across threshold range
7. **Feature Correlation**: Which features correlate with anomaly scores

**Design Principle**: Each plot answers a specific question about model behavior.

## Explainability with SHAP

### Two-Level Explanation Strategy

**Global Explainability** (SHAP Summary Plot):
- Which features are most important overall?
- How do feature values influence predictions?
- Ranked feature importance across all samples

**Local Explainability** (SHAP Waterfall Plots):
- Why was this specific sample flagged?
- Which features contributed most to this prediction?
- Generated for top 5 highest-scoring anomalies

### SHAP Implementation Choices

- **Tree-based models** (Isolation Forest): Use `TreeExplainer` (fast, exact)
- **Linear/KNN models**: Use `KernelExplainer` (slower, approximate)
- **Sample Limit**: Default 100 samples to balance insight vs compute time
- **Output Formats**: Visual plots (PNG) + CSV export for further analysis

## Dataset-Specific Adaptations

### IMS (Unsupervised Time-Series)
- **Focus**: Temporal progression of anomaly scores
- **Expected Pattern**: Low scores (healthy) → rising scores → spike (failure)
- **Key Visualization**: Timeline plot showing degradation trajectory
- **SHAP Insight**: Which vibration features (RMS, kurtosis, peak-to-peak) drive failures

### CWRU (Supervised Multi-Class)
- **Focus**: Distinguishing fault types
- **Key Visualization**: Confusion matrix showing per-fault performance
- **SHAP Insight**: Feature signatures that distinguish ball faults vs inner/outer race faults

### AI4I (Supervised Imbalanced)
- **Focus**: Balancing false alarms vs missed failures
- **Key Visualization**: Precision-Recall curve (better than ROC for imbalance)
- **SHAP Insight**: Which process parameters (temperature, torque, RPM) predict failure modes

### C-MAPSS (Regression to Anomaly)
- **Focus**: Correlation between anomaly scores and Remaining Useful Life (RUL)
- **Key Visualization**: Timeline showing score increase as RUL decreases
- **Additional Metric**: Pearson/Spearman correlation between scores and RUL
- **SHAP Insight**: Which sensor readings indicate imminent failure

## HTML Report Structure

The final deliverable is a comprehensive, self-contained HTML report:

1. **Executive Summary**: Key metrics at a glance (anomaly rate, model type, threshold)
2. **Performance Metrics**: Tables organized by metric type
3. **Visualizations**: All plots embedded as base64 images
4. **Explainability**: SHAP plots with interpretation guidance
5. **Configuration**: Full config snapshot for reproducibility
6. **Run Metadata**: Git commit, timestamp, software versions

**Design Goal**: Report should be shareable with stakeholders who don't have technical environment access.

## Error Handling Philosophy

**Graceful Degradation**: System continues with reduced functionality rather than failing completely.

- **Missing Labels**: Skip supervised metrics, generate unsupervised metrics only
- **SHAP Errors**: Catch exceptions, continue with other outputs, log warning
- **Plot Failures**: Skip individual failed plots, generate remaining visualizations
- **Memory Issues**: Automatically reduce SHAP sample size

**Validation**: Pre-flight checks verify all required files exist before starting expensive computation.

## Performance Considerations

**Target**: Complete evaluation in < 5 minutes per model on laptop hardware

**Optimization Strategies**:
- SHAP limited to 100 samples by default (configurable)
- TreeExplainer for tree-based models (10-100x faster than KernelExplainer)
- Vectorized plot generation where possible
- Parquet format for fast I/O

**Memory Management**:
- Load features in chunks if dataset exceeds 1M samples
- Clear intermediate results after visualization generation

## Success Criteria

A successful evaluate.py implementation should:

1. Work with all 10 trained models without code modification
2. Generate meaningful insights for both supervised and unsupervised scenarios
3. Produce publication-quality visualizations
4. Explain individual predictions with SHAP
5. Create comprehensive, shareable HTML reports
6. Log all metadata for full reproducibility
7. Handle errors gracefully without crashing
8. Complete in < 5 minutes per model on laptop

## Next Steps

**Immediate**: Implement evaluate.py following this design

**Subsequent Stages**:
- **Stage 6**: score_batch.py for production inference
- **Dashboards**: Streamlit app for interactive exploration
- **Model Comparison**: Cross-model analysis to select best performer per dataset
- **Deployment**: Package best models for production use

## Design Rationale

### Why HTML Reports?
- Self-contained (no dependencies to view)
- Shareable with non-technical stakeholders
- Embeds all visualizations inline
- Version-controllable for historical comparison

### Why SHAP over Other Methods?
- Model-agnostic (works with all our models)
- Theoretically grounded (game theory)
- Both global and local explanations
- Industry standard for ML explainability

### Why Dataset-Adaptive Metrics?
- Different evaluation paradigms (supervised vs unsupervised)
- Inappropriate metrics mislead (e.g., accuracy on imbalanced data)
- Stakeholders need metrics relevant to their domain

### Why Separate Threshold from Model?
- Same model can be tuned for different false-alarm tolerances
- Threshold calibration is business decision, not technical one
- Allows sensitivity analysis without retraining
