# Feature Extraction Guide

## Overview

The `make_features.py` script is **Stage 2** of the predictive maintenance pipeline. It transforms cleaned sensor data into engineered features suitable for machine learning.

**Pipeline Flow:**
```
prepare → features → train → threshold → evaluate → score
   ↑         ↑
 Stage 1   Stage 2 (you are here)
```

## Quick Start

### Basic Usage

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

## What It Does

### For Windowed Datasets (IMS, CWRU, C-MAPSS)

1. **Loads cleaned data** from `data/clean/<dataset>/<dataset>_clean.parquet`
2. **Creates rolling windows** from time-series signals
   - Window size and stride defined in config (e.g., 2048 samples, 1024 stride)
   - Overlapping windows for better coverage
3. **Computes time-domain features** for each window:
   - **Statistical**: mean, std, min, max, median
   - **Signal**: rms, peak_to_peak, crest_factor
   - **Shape**: kurtosis, skewness
4. **Preserves metadata** (timestamps, fault labels, etc.)
5. **Saves features** to `data/features/<dataset>/` in both Parquet and CSV

### For Tabular Datasets (AI4I)

1. **Loads cleaned data**
2. **Adds derived features** (e.g., temperature difference, power)
3. **Passes through** existing columns
4. **Saves to features directory**

## Configuration

Features are defined in the dataset YAML config:

```yaml
schema:
  computed_features:
    - rms                # Root Mean Square (energy)
    - peak_to_peak       # Full amplitude range
    - kurtosis           # Tail heaviness (excess kurtosis)
    - skewness           # Asymmetry
    - mean               # Average
    - std                # Standard deviation
    - max                # Maximum value
    - min                # Minimum value
    # - fft_band_energy  # Not yet implemented

prep:
  use_windowing: true    # Set to false for tabular data (AI4I)
  window:
    size: 2048          # Samples per window
    stride: 1024        # Step between windows (50% overlap)
    overlap: true
```

## Feature Naming Convention

Features follow the pattern: `{channel}_{feature_name}`

**Examples:**
- `ch1_rms` - RMS of channel 1 (IMS)
- `ch2_kurtosis` - Kurtosis of channel 2 (IMS)
- `vibration_peak_to_peak` - Peak-to-peak amplitude (CWRU)
- `s1_mean` - Mean of sensor 1 (C-MAPSS)

## Output Format

### Parquet (Primary)
- **Location**: `data/features/<dataset>/<dataset>_features.parquet`
- **Compression**: Snappy
- **Use case**: Production pipeline (faster I/O, smaller size)

### CSV (Inspection)
- **Location**: `data/features/<dataset>/<dataset>_features.csv`
- **Use case**: Human inspection, debugging, external tools

### Metadata Log
- **Location**: `data/features/<dataset>/feature_extraction_log.json`
- **Contains**: Timestamp, config, statistics, feature list

## Dataset-Specific Examples

### IMS (8-Channel Vibration)

**Input:**
- 96 files × ~20,480 samples/file
- 8 channels (ch1-ch8)

**Processing:**
- Each file split into ~19 windows (size=2048, stride=1024)
- Features computed per channel per window
- ~14,000 feature vectors created

**Output columns:**
```
timestamp, file_index, source_file, window_id, sensor,
ch1_mean, ch1_std, ch1_rms, ch1_kurtosis, ch1_skewness, ch1_peak_to_peak,
ch2_mean, ch2_std, ch2_rms, ...
```

### CWRU (Fault Classification)

**Input:**
- Multiple CSV files with vibration_amp
- Fault labels extracted from filenames

**Processing:**
- Each recording split into windows
- Features preserve fault metadata

**Output columns:**
```
source_file, fault_type, fault_size_mils, load_hp, sensor_location, window_id,
vibration_mean, vibration_std, vibration_rms, vibration_kurtosis, ...
```

### AI4I (Tabular Manufacturing)

**Input:**
- Single CSV with process parameters

**Processing:**
- No windowing (use_windowing: false)
- Adds derived features

**Output columns:**
```
type, air_temp_k, proc_temp_k, rpm, torque_nm, tool_wear_min,
temp_diff, power,  # Derived features
target, twf, hdf, pwf, osf, rnf  # Target labels
```

### C-MAPSS (Turbofan Degradation)

**Input:**
- 21 sensors (s1-s21) per engine unit
- Multiple engines with cycle-by-cycle data

**Processing:**
- Windows of 30 cycles per sensor
- Grouped by unit to prevent leakage

**Output columns:**
```
unit, cycle, split, setting1, setting2, setting3, window_id, sensor,
s1_mean, s1_std, s1_rms, s1_kurtosis, ...
```

## Feature Descriptions

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **mean** | `np.mean(x)` | Average amplitude |
| **std** | `np.std(x)` | Signal variability |
| **rms** | `sqrt(mean(x²))` | Energy content (increases with degradation) |
| **min/max** | `np.min(x)`, `np.max(x)` | Range boundaries |
| **peak_to_peak** | `max - min` | Full amplitude span |
| **kurtosis** | `scipy.stats.kurtosis(x, fisher=True)` | Tail heaviness (excess kurtosis, 0=normal) |
| **skewness** | `scipy.stats.skew(x)` | Distribution asymmetry (0=symmetric) |
| **crest_factor** | `max(|x|) / rms` | Peak sharpness (high = impulsive faults) |

### Why These Features?

**Vibration analysis principles:**
- **RMS** increases as bearing degrades (more energy dissipated)
- **Kurtosis** spikes when faults create impulsive events
- **Crest factor** detects sharp peaks (ball bearing defects)
- **Peak-to-peak** shows amplitude growth over time

## Memory Efficiency

The script is designed to run on laptops (2-4 GB RAM):

- **Chunked processing**: Processes files one at a time
- **Generator patterns**: Windows created on-the-fly
- **Parquet format**: Columnar storage reduces memory overhead
- **Efficient numpy ops**: Vectorized computations

## Validation

The script automatically validates outputs:

1. **NaN check**: Warns if features contain NaN, fills with 0.0
2. **Type check**: Verifies numeric columns are numeric
3. **Logging**: Outputs statistics and warnings

## Troubleshooting

### Issue: "Clean data not found"

**Solution:**
```bash
# Run Stage 1 first
python scripts/prep_data.py --config configs/ims.yaml
```

### Issue: "Memory Error"

**Solution:**
- Reduce max_files in config (test with subset)
- Increase stride to reduce window count
- Process datasets individually

### Issue: "All NaN features"

**Cause:** Window size larger than signal length

**Solution:**
- Reduce window size in config
- Check if clean data has sufficient samples

### Issue: "Unknown dataset type"

**Cause:** Config missing required path keys

**Solution:**
- Verify config has one of: `raw_input_dir`, `raw_input_dirs`, `raw_input_file`, `raw_train_file`

## Next Steps

After feature extraction, proceed to **Stage 3: Training**

```bash
python scripts/train.py --config configs/models/isolation_forest.yaml
```

See [TRAINING.md](TRAINING.md) for details.

## Performance Benchmarks

Approximate execution times on 2 vCPU laptop:

| Dataset | Input Size | Output Size | Time | Memory |
|---------|-----------|-------------|------|--------|
| IMS (30 files) | ~600K rows | ~14K features | ~30s | ~500 MB |
| CWRU | ~1.2M rows | ~25K features | ~45s | ~800 MB |
| AI4I | 10K rows | 10K rows | <5s | ~50 MB |
| C-MAPSS FD001 | ~20K rows | ~50K features | ~60s | ~600 MB |

## Implementation Notes

### Window Calculation

For a signal of length `N` with window size `W` and stride `S`:

```python
num_windows = (N - W) // S + 1
```

Example:
- N = 20,480 samples
- W = 2048 samples
- S = 1024 samples
- Windows = (20,480 - 2048) // 1024 + 1 = 19 windows

### Feature Count

Total feature columns = (number of channels) × (number of features)

**IMS example:**
- 8 channels × 8 features = 64 feature columns
- Plus metadata: timestamp, file_index, source_file, window_id, sensor
- Total: ~69 columns

### Overlap Benefits

50% overlap (stride = window_size / 2):
- **More samples**: 2x more training data
- **Smoother transitions**: Reduces edge effects
- **Better anomaly detection**: Less likely to miss events

### Fisher's Kurtosis (Excess Kurtosis)

The script uses `fisher=True` which computes excess kurtosis:
- **0** = normal distribution
- **> 3** = heavy tails (leptokurtic) - common in faults
- **< 0** = light tails (platykurtic)

This is standard in anomaly detection vs. Pearson's kurtosis (fisher=False) which has baseline of 3.

## Advanced Usage

### Custom Feature Functions

To add new features, edit `compute_time_domain_features()`:

```python
if 'custom_feature' in feature_list:
    features['custom_feature'] = your_function(clean_data)
```

Then add to config:
```yaml
computed_features:
  - custom_feature
```

### FFT Band Energy (Future)

Currently a placeholder. To implement:

1. Add sampling rate to config
2. Define frequency bands
3. Implement FFT computation in `compute_time_domain_features()`

Example:
```python
from scipy.fft import rfft, rfftfreq

fft_vals = np.abs(rfft(clean_data))
freqs = rfftfreq(len(clean_data), 1/sampling_rate)

# Low band: 0-1000 Hz
mask = (freqs >= 0) & (freqs < 1000)
features['fft_low_band'] = np.sum(fft_vals[mask]**2)
```

## References

- **Time-domain features**: ISO 10816 (Vibration monitoring)
- **Statistical features**: NIST/SEMATECH e-Handbook of Statistical Methods
- **Kurtosis**: Fisher (1930), "The moments of the distribution for normal samples"
- **Vibration analysis**: Randall, R.B. (2011), "Vibration-based Condition Monitoring"
