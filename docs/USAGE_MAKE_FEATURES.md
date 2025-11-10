# Feature Extraction Quick Start Guide

## Overview

`scripts/make_features.py` is **Stage 2** of the predictive maintenance pipeline. It transforms cleaned sensor data into machine learning features.

## Prerequisites

1. **Python environment** with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Cleaned data** from Stage 1:
   ```bash
   # Run Stage 1 first if you haven't
   python scripts/prep_data.py --config configs/ims.yaml
   ```

## Basic Usage

### Run Feature Extraction

```bash
# IMS bearing dataset
python scripts/make_features.py --config configs/ims.yaml

# CWRU bearing faults
python scripts/make_features.py --config configs/cwru.yaml

# AI4I manufacturing
python scripts/make_features.py --config configs/ai4i.yaml

# C-MAPSS turbofan
python scripts/make_features.py --config configs/fd001.yaml
```

### Expected Output

The script will:
1. Load cleaned data from `data/clean/<dataset>/`
2. Create rolling windows (for time-series datasets)
3. Compute time-domain features
4. Save to `data/features/<dataset>/`:
   - `<dataset>_features.parquet` (production format)
   - `<dataset>_features.csv` (human-readable)
   - `feature_extraction_log.json` (metadata)

### Example Output

```
2025-11-09 16:30:00 - INFO - Loading configuration from configs/ims.yaml
2025-11-09 16:30:00 - INFO - Dataset: ims
2025-11-09 16:30:00 - INFO - Loading cleaned data from data/clean/ims/ims_clean.parquet...
2025-11-09 16:30:01 - INFO - Loaded 1,966,080 rows with 11 columns
2025-11-09 16:30:01 - INFO - Extracting features for IMS dataset
2025-11-09 16:30:01 - INFO - Sensor columns: ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
2025-11-09 16:30:01 - INFO - Window size: 2048, stride: 1024
2025-11-09 16:30:01 - INFO - Features to compute: ['rms', 'peak_to_peak', 'kurtosis', 'skewness', 'mean', 'std', 'max', 'min']
Processing files: 100%|██████████| 96/96 [00:25<00:00,  3.76it/s]
2025-11-09 16:30:27 - INFO - Created 14,592 feature vectors
2025-11-09 16:30:27 - INFO - Validation complete
2025-11-09 16:30:27 - INFO - Saving features to data/features/ims/ims_features.parquet (Parquet)...
2025-11-09 16:30:28 - INFO - Saved 14,592 rows to Parquet (3.45 MB)
2025-11-09 16:30:28 - INFO - Saving features to data/features/ims/ims_features.csv (CSV)...
2025-11-09 16:30:29 - INFO - Saved 14,592 rows to CSV (8.92 MB)
================================================================================
Feature extraction complete!
Total feature vectors: 14,592
Total columns: 69
================================================================================
```

## Inspect Results

### View Features (Command Line)

```bash
# First 5 rows
head -5 data/features/ims/ims_features.csv

# Column names
head -1 data/features/ims/ims_features.csv | tr ',' '\n'

# Check file size
ls -lh data/features/ims/
```

### View Features (Python)

```python
import pandas as pd

# Load features
df = pd.read_parquet('data/features/ims/ims_features.parquet')

# Basic info
print(df.shape)
print(df.columns)
print(df.head())

# Feature statistics
feature_cols = [c for c in df.columns if 'rms' in c or 'kurtosis' in c]
print(df[feature_cols].describe())
```

### Use Demo Script

```bash
python examples/feature_extraction_demo.py ims
```

## Configuration

Edit `configs/<dataset>.yaml` to customize feature extraction:

```yaml
schema:
  computed_features:
    - rms                # Root Mean Square
    - peak_to_peak       # Amplitude range
    - kurtosis           # Tail heaviness
    - skewness           # Asymmetry
    - mean               # Average
    - std                # Standard deviation
    - max                # Maximum
    - min                # Minimum

prep:
  use_windowing: true    # Set to false for tabular data
  window:
    size: 2048          # Samples per window
    stride: 1024        # Step between windows (50% overlap)
    overlap: true
```

## Features Computed

| Feature | Description | Use Case |
|---------|-------------|----------|
| **rms** | Root Mean Square energy | Increases with bearing degradation |
| **peak_to_peak** | Max - Min amplitude | Shows amplitude growth |
| **kurtosis** | Tail heaviness (excess) | Spikes during impulsive faults |
| **skewness** | Distribution asymmetry | Detects bias in vibration |
| **mean** | Average amplitude | Baseline shift detection |
| **std** | Standard deviation | Variability measure |
| **max/min** | Range boundaries | Extreme value tracking |
| **crest_factor** | Peak sharpness | Ball bearing defect indicator |

## Output Schema

### IMS Dataset

```
Columns:
  - timestamp          (datetime)
  - file_index         (int)
  - source_file        (string)
  - window_id          (int)
  - sensor             (string)
  - ch1_mean           (float)
  - ch1_std            (float)
  - ch1_rms            (float)
  - ch1_kurtosis       (float)
  - ch1_skewness       (float)
  - ch1_peak_to_peak   (float)
  - ch1_max            (float)
  - ch1_min            (float)
  - ... (same for ch2-ch8)

Rows: ~14,000 (for 96 files with 2048 window, 1024 stride)
```

### CWRU Dataset

```
Columns:
  - source_file           (string)
  - fault_type            (string: normal, ball_fault, inner_race_fault, outer_race_fault)
  - fault_size_mils       (int)
  - load_hp               (int)
  - sensor_location       (string)
  - window_id             (int)
  - vibration_mean        (float)
  - vibration_std         (float)
  - vibration_rms         (float)
  - vibration_kurtosis    (float)
  - ... (other features)

Rows: ~25,000 (varies by file count)
```

### AI4I Dataset

```
Columns:
  - type              (string: H, L, M)
  - air_temp_k        (float)
  - proc_temp_k       (float)
  - rpm               (float)
  - torque_nm         (float)
  - tool_wear_min     (float)
  - temp_diff         (float, derived)
  - power             (float, derived)
  - target            (int: 0 or 1)
  - twf, hdf, pwf, osf, rnf  (int: fault types)

Rows: 10,000 (same as input - no windowing)
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: FileNotFoundError

**Error:**
```
FileNotFoundError: Clean data not found: data/clean/ims/ims_clean.parquet
```

**Solution:**
```bash
# Run Stage 1 first
python scripts/prep_data.py --config configs/ims.yaml
```

### Issue: Memory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
1. Reduce `max_files` in config (test with subset):
   ```yaml
   paths:
     max_files: 30  # Instead of null (all files)
   ```

2. Increase `stride` to reduce window count:
   ```yaml
   window:
     stride: 2048  # Instead of 1024 (no overlap)
   ```

### Issue: All NaN Features

**Warning:**
```
WARNING - Found NaN values in columns: {'ch1_kurtosis': 100}
```

**Cause:** Window size larger than signal segments

**Solution:**
- Reduce window size in config (try 512 or 1024)
- Check input data has sufficient samples per file

## Next Steps

After feature extraction, proceed to **Stage 3: Model Training**

```bash
# Train Isolation Forest model
python scripts/train.py --config configs/models/isolation_forest.yaml
```

## File Locations

### Input
- **Cleaned data**: `data/clean/<dataset>/<dataset>_clean.parquet`
- **Configuration**: `configs/<dataset>.yaml`

### Output
- **Features (Parquet)**: `data/features/<dataset>/<dataset>_features.parquet`
- **Features (CSV)**: `data/features/<dataset>/<dataset>_features.csv`
- **Log**: `data/features/<dataset>/feature_extraction_log.json`

## Performance

Approximate execution times (2 vCPU laptop):

| Dataset | Input Rows | Output Rows | Time | Memory |
|---------|-----------|-------------|------|--------|
| IMS (30 files) | 600K | 14K | ~30s | ~500 MB |
| IMS (96 files) | 2M | 14K | ~60s | ~800 MB |
| CWRU | 1.2M | 25K | ~45s | ~800 MB |
| AI4I | 10K | 10K | <5s | ~50 MB |
| FD001 | 20K | 50K | ~60s | ~600 MB |

## Additional Resources

- **Detailed documentation**: `docs/FEATURE_EXTRACTION.md`
- **Unit tests**: `tests/test_make_features.py`
- **Demo script**: `examples/feature_extraction_demo.py`
- **Project guide**: `CLAUDE.md`

## Getting Help

1. Check logs in `data/features/<dataset>/feature_extraction_log.json`
2. Run demo script to inspect output
3. Check unit tests: `pytest tests/test_make_features.py -v`
4. Review config file for correct settings

## Quick Reference

```bash
# Full pipeline sequence
python scripts/prep_data.py --config configs/ims.yaml
python scripts/make_features.py --config configs/ims.yaml
python scripts/train.py --config configs/models/isolation_forest.yaml

# Inspect results
python examples/feature_extraction_demo.py ims

# Run tests
pytest tests/test_make_features.py -v
```
