# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a predictive maintenance pipeline for anomaly detection on machine sensor data. It's CPU-only, config-driven, and designed to run on laptops. The system uses unsupervised learning to detect anomalies in time-series vibration data from bearings and other industrial equipment.

## Development Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### Running Tests
```bash
pytest -q
```

### Data Processing Pipeline

The pipeline follows this sequence: **prepare → features → train → threshold → evaluate → score**

1. **Data Preparation** (clean raw data):
```bash
python scripts/prep_data.py --config configs/ims.yaml
# Or for other datasets:
# python scripts/prep_data.py --config configs/cwru.yaml
# python scripts/prep_data.py --config configs/ai4i.yaml
# python scripts/prep_data.py --config configs/fd001.yaml
```

2. **Feature Engineering** (create rolling window features):
```bash
python scripts/make_features.py --config configs/ims.yaml
```

3. **Training** (train anomaly detection model):
```bash
python scripts/train.py --config configs/models/isolation_forest.yaml
# Or: configs/models/knn_lof.yaml, configs/models/one_class_svm.yaml, configs/models/autoencoder.yaml
```

4. **Threshold Setting** (calibrate alert threshold to target false-alarm rate):
```bash
python scripts/threshold.py --target_far 0.1/week
```

5. **Evaluation** (generate metrics and reports):
```bash
python scripts/evaluate.py --report artifacts/reports/ims_iforest/
```

6. **Batch Scoring** (score new data):
```bash
python scripts/score_batch.py --config configs/ims.yaml \
  --model artifacts/models/ims_iforest.joblib \
  --input data/processed/ims/test.csv \
  --output artifacts/reports/ims_iforest/scores.csv
```

### Dashboard (Optional)
```bash
streamlit run dashboards/app.py
```

## Architecture

### Configuration-Driven Design

The system is heavily config-driven using YAML files. **Edit configs, not code.**

- **Dataset configs** (`configs/*.yaml`): Define data loading, preprocessing, windowing, and splitting
  - Each config specifies paths, schema (column mapping), file format, windowing parameters, and train/val/test splits
  - Supports multiple dataset types: IMS (time-series vibration), CWRU (bearing faults), AI4I (tabular), C-MAPSS (turbofan degradation)

- **Model configs** (`configs/models/*.yaml`): Define model type, hyperparameters, scaler, and features to use

### Data Flow

1. **Raw Data** (`data/raw/`): Original downloaded files (vibration signals, CSV files, etc.)
2. **Clean Data** (`data/clean/`): Loaded and normalized data with timestamps and metadata
3. **Features** (`data/features/`): Rolling-window features extracted from signals
4. **Models** (`results/models/`): Trained anomaly detection models (saved as .joblib)
5. **Reports** (`results/reports/`): Metrics, plots, SHAP explanations, run logs

### Supported Datasets

- **IMS**: Time-series vibration from bearings run until failure (unsupervised)
- **CWRU**: Bearing fault classification with labeled fault types
- **AI4I**: Tabular manufacturing data with failure modes
- **C-MAPSS**: Turbofan engine degradation (FD001-FD004)

### Feature Engineering

Time-domain features computed on rolling windows:
- Statistical: mean, std, RMS, peak-to-peak, min, max
- Shape: kurtosis, skewness, crest factor
- Optional: FFT frequency band energies

Window configuration (in dataset YAML):
- `size`: samples per window (e.g., 2048)
- `stride`: step between windows (e.g., 1024 for 50% overlap)
- `overlap`: boolean to enable/disable overlapping windows

### Models

- **Isolation Forest**: Fast, works well for high-dimensional data
- **kNN-LOF**: Local Outlier Factor using k-nearest neighbors
- **One-Class SVM**: Boundary-based anomaly detection
- **Autoencoder** (optional): Requires PyTorch (uncomment in requirements.txt)

All models use scikit-learn except Autoencoder.

### Anomaly Detection Pattern

For time-series datasets (IMS, C-MAPSS):
- Early data = "normal" baseline (healthy equipment)
- Training fits on normal data only
- Anomaly scores flag deviation from normal
- Config parameter: `normal_baseline_files` or `normal_baseline_time_hours`

### Key Scripts

- `scripts/prep_data.py`: Main data loader supporting all 4 dataset types
  - Auto-detects dataset type from config paths (`raw_input_dir` = IMS, `raw_input_dirs` = CWRU, `raw_input_file` = AI4I, `raw_train_file` = C-MAPSS)
  - Handles timestamp extraction from filenames (IMS format: `YYYY.MM.DD.HH.MM.SS`)
  - Extracts metadata from filenames (CWRU fault types, sizes, loads)
  - Saves both Parquet (compressed, fast) and CSV formats

- `scripts/convert_cwru_mat_to_csv.py`: Converts MATLAB .mat files to CSV for CWRU dataset

### Reproducibility

- All runs log git commit, config, and random seeds to `artifacts/reports/<run>/run.json`
- Fixed random seeds in configs (`random_state: 42`)
- Parquet format preserves data types exactly

### Explainability

- SHAP (SHapley Additive exPlanations) generates feature importance
- Top drivers identified per alert
- Summaries saved in reports directory

## Important Non-Obvious Relationships

### Config-Based Dataset Detection

Scripts auto-detect dataset type from config path keys (no explicit type flag needed):
- `raw_input_dir` → IMS loader
- `raw_input_dirs` → CWRU loader
- `raw_input_file` → AI4I loader
- `raw_train_file` → C-MAPSS loader

### Filename Metadata Extraction

Dataset loaders extract critical metadata from filenames:
- **IMS**: Timestamp from filename (`2003.10.22.12.06.24` → datetime)
- **CWRU**: Fault type, size, load, sensor from filename (`B007_0__X118_DE_time.csv` → ball fault, 0.007", load 0hp, drive-end sensor)

This metadata becomes training labels or temporal indices.

### Windowing Multiplies Samples

One raw file generates many windowed samples:
- Example: 20,480 samples with window=2048, stride=1024 → ~19 windows
- 96 files × 19 windows × 8 channels = ~14,592 feature vectors

### Normal Baseline Training (IMS)

For IMS, "training" means learning the distribution of healthy operation:
- Only first N files used for model fitting (`normal_baseline_files`)
- Later files used exclusively for anomaly scoring
- This is unsupervised learning, not supervised classification

### Dual Output Formats

All pipeline stages save both formats:
- **Parquet**: 10x smaller, preserves types, faster I/O (production)
- **CSV**: Human-readable, universally compatible (inspection)

Never remove CSV output even if it seems redundant.

### Threshold as Post-Processing

Anomaly detection separates model training from alert tuning:
1. Model outputs continuous anomaly score
2. Threshold converts score to binary alert
3. Threshold calibrated separately to target false-alarm rate
4. Allows sensitivity adjustment without retraining

### Data Splitting Strategies

Four splitting methods based on dataset characteristics:

- **time_based** (IMS): Chronological split prevents leakage in sequential data
  - Early files → train (healthy)
  - Middle files → validation (transition)
  - Late files → test (degraded/failed)

- **random_percent** (CWRU, AI4I): Stratified random split maintains class balance
  - Requires `stratify_by` column

- **by_unit_holdout** (C-MAPSS): Split by entire engine units
  - All cycles from one unit stay together
  - Prevents leakage between same equipment
  - Uses `group_by: unit`

- **by_file** (CWRU alternative): All windows from one file in same split

### DVC Integration

Data versioning requires TWO pushes:
```bash
dvc add data/raw/ims/1st_test      # Track with DVC
git add data/raw/ims/1st_test.dvc  # Add pointer to Git
git commit -m "Update IMS data"
dvc push                            # Upload data to DVC remote
git push                            # Upload pointer to Git
```

Team members sync with: `git pull && dvc pull`

## Configuration Structure

### Dataset Config (configs/*.yaml)

```yaml
dataset_name: <name>

paths:
  raw_input_dir/file/dirs: <source>
  clean_output_path: <parquet>
  clean_output_path_csv: <csv>
  features_output_path: <parquet>
  features_output_path_csv: <csv>

schema:
  column_map: {old_name: new_name}
  target_col: <label column>
  drop_features: [list]
  categorical_features: [list]
  numeric_features: [list]
  computed_features: [rms, peak_to_peak, kurtosis, ...]

prep:
  drop_na: true/false
  impute: mean/median/mode
  scale: standard/minmax
  use_windowing: true/false
  window:
    size: 2048
    stride: 1024
    overlap: true/false
  anomaly_detection:  # IMS-specific
    use_early_as_normal: true
    normal_baseline_files: 20

split:
  method: time_based/random_percent/by_unit_holdout/by_file
  train_ratio: 0.60
  val_ratio: 0.10
  test_ratio: 0.30
  stratify_by: <column or null>
  random_state: 42
  group_by: <column>  # CMAPSS-specific

file_format:
  delimiter: "\t"
  header: true/false
  columns_in_order: [list]
  extract_timestamp_from_filename: true/false
  timestamp_format: "%Y.%m.%d.%H.%M.%S"
```

### Model Config (configs/models/*.yaml)

```yaml
model_name: <name>
model_type: isolation_forest/knn_lof/one_class_svm/autoencoder
dataset_config: configs/ims.yaml

hyperparameters:
  # Isolation Forest
  n_estimators: 100
  contamination: 0.1
  max_features: 1.0

  # kNN-LOF
  n_neighbors: 20
  contamination: 0.1

  # One-Class SVM
  kernel: rbf
  nu: 0.1
  gamma: auto

scaler: standard/minmax/robust
features: [list or 'all']
random_state: 42
```

## Implementation Status

**COMPLETED:**
- Data preparation (prep_data.py) for all 4 datasets
- MATLAB-to-CSV converter (convert_cwru_mat_to_csv.py)
- Complete configuration system (7 dataset configs + 4 model configs)
- DVC setup for data versioning
- All datasets fully cleaned (ims_clean, cwru_clean, ai4i_clean, fd001-004_clean)
- Feature engineering (make_features.py) for all 4 datasets
- Model training (train.py) for all datasets with multiple models
- Threshold calibration (threshold.py) for 10 trained models
- Model configuration YAMLs (isolation_forest, knn_lof, one_class_svm, autoencoder)

**PLANNED:**
- scripts/evaluate.py (Stage 5)
- scripts/score_batch.py (Stage 6)
- dashboards/app.py (Streamlit)
- Test suite (pytest)

## Development Guidelines

### When to Edit Configs vs. Code

**Edit configs when:**
- Changing dataset paths or file formats
- Adjusting window size, stride, or overlap
- Modifying train/val/test split ratios
- Tuning model hyperparameters
- Adding/removing features
- Changing preprocessing steps

**Edit code when:**
- Adding a new dataset type
- Implementing a new feature extraction method
- Adding a new model architecture
- Creating new evaluation metrics
- Building new pipeline stages

### Dataset-Specific Considerations

**IMS (Unsupervised):**
- Must set `normal_baseline_files` (typically 20-30)
- Use `time_based` splitting
- No labels required
- Focus on threshold calibration

**CWRU (Supervised):**
- Requires fault labels in filenames
- Use `random_percent` with `stratify_by: fault_type`
- Classification metrics (accuracy, F1, confusion matrix)

**AI4I (Supervised Tabular):**
- No windowing needed (`use_windowing: false`)
- Multiple binary failure modes available
- Simple tabular ML problem

**C-MAPSS (Regression):**
- RUL (Remaining Useful Life) prediction
- Use `by_unit_holdout` splitting with `group_by: unit`
- RUL capping at 125 cycles prevents early-life dominance

### CPU Constraints

System designed for 2-4 GB RAM, ~2 vCPU:
- Prefer scikit-learn over PyTorch
- Keep ensemble sizes small (100 trees, not 1000)
- Use Parquet for fast I/O
- No GPU required

### Reproducibility Requirements

Always include in pipeline runs:
- Git commit hash
- Full config snapshot
- Random seeds (set to 42 by default)
- Timestamp
- Save to `results/reports/<run>/run.json`

### Common Pitfalls

1. **Don't train on degraded data (IMS)**: Only use `normal_baseline_files` for training
2. **Don't skip stratification (supervised)**: CWRU and AI4I need `stratify_by` to prevent class imbalance
3. **Don't mix dataset types**: Each config is single-dataset; run separately
4. **Don't forget window overlap effects**: Overlapping windows create correlation between samples
5. **Don't skip intermediate inspection**: Always check clean/ outputs before features, features/ before training
6. **Don't hardcode paths**: Always read from config, use pathlib for cross-platform compatibility
7. **Don't skip DVC for large files**: Raw data goes in DVC, not Git
## Current Status

**Completed Stages:**
1. ✅ Stage 1: Data Preparation (prep_data.py) - All datasets cleaned
2. ✅ Stage 2: Feature Engineering (make_features.py) - Features extracted for all datasets
3. ✅ Stage 3: Model Training (train.py) - 10 models trained across all datasets
4. ✅ Stage 4: Threshold Calibration (threshold.py) - All models calibrated with optimal thresholds

### Threshold Calibration Results (Stage 4)

All 10 models have been successfully calibrated with their optimal thresholds:

#### IMS Dataset (4 models)
| Model | Target FAR | Estimated FAR | Threshold | Status |
|-------|------------|---------------|-----------|--------|
| Isolation Forest | 1.0/week | 0.989/week | 0.4851 | ✅ Excellent (99% accuracy) |
| AutoEncoder | 0.2/week | 0.200/week | 0.0137 | ✅ Perfect |
| KNN LOF | 0.2/week | 0.200/week | 1.7821 | ✅ Perfect |
| One-Class SVM | 2.0/week | 2.000/week | -0.3182 | ✅ Perfect (fixed with nu=0.3) |

#### AI4I Dataset (1 model)
| Model | Target FAR | Estimated FAR | Threshold | Status |
|-------|------------|---------------|-----------|--------|
| Isolation Forest | 0.2/week | 0.202/week | 0.4863 | ✅ Excellent |

#### CWRU Dataset (1 model)
| Model | Target FAR | Estimated FAR | Threshold | Status |
|-------|------------|---------------|-----------|--------|
| Isolation Forest | 0.2/week | 0.212/week | 0.4795 | ✅ Good |

#### NASA C-MAPSS Dataset (4 subsets)
| Model | Target FAR | Estimated FAR | Threshold | Status |
|-------|------------|---------------|-----------|--------|
| FD001 Isolation Forest | 0.2/week | 0.289/week | 0.4912 | ✅ Good (slightly high) |
| FD002 Isolation Forest | 0.2/week | 0.216/week | 0.5025 | ✅ Good |
| FD003 Isolation Forest | 0.2/week | 0.217/week | 0.4933 | ✅ Good |
| FD004 Isolation Forest | 0.2/week | 0.221/week | 0.5016 | ✅ Good |

**Key Insights:**
- All 10 models successfully calibrated with FAR estimates within acceptable range
- IMS models show excellent calibration accuracy (perfect for 3/4 models)
- AutoEncoder and KNN LOF achieved perfect FAR matching (0.200/week target)
- One-Class SVM required hyperparameter adjustment (nu=0.3) for stable threshold calibration
- NASA C-MAPSS FD001 shows slightly higher FAR (0.289 vs 0.2 target) but still acceptable

**Next Steps:**
- Implement Stage 5: Evaluation (evaluate.py) - Generate comprehensive metrics, plots, and SHAP explanations