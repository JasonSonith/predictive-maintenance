# Feature Extraction Testing Checklist

Use this checklist to verify `scripts/make_features.py` is working correctly.

## Prerequisites

- [ ] Python environment with dependencies installed
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Stage 1 (data preparation) completed
  ```bash
  ls -la data/clean/ims/ims_clean.parquet      # Should exist
  ls -la data/clean/cwru/cwru_clean.parquet    # Should exist
  ls -la data/clean/ai4i/ai4i_clean.parquet    # Should exist
  ls -la data/clean/cmapss/fd001_clean.parquet # Should exist
  ```

## Basic Functionality Tests

### Test 1: IMS Dataset (Multi-Channel Vibration)

- [ ] Run feature extraction
  ```bash
  python scripts/make_features.py --config configs/ims.yaml
  ```

- [ ] Verify outputs created
  ```bash
  ls -la data/features/ims/ims_features.parquet
  ls -la data/features/ims/ims_features.csv
  ls -la data/features/ims/feature_extraction_log.json
  ```

- [ ] Check output shape
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); print(f'Shape: {df.shape}'); print(f'Columns: {df.columns.tolist()}')"
  ```
  - Expected: ~14,000 rows (depends on max_files setting)
  - Expected: ~69 columns (metadata + 8 channels × 8 features)

- [ ] Verify no NaN values
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); print(f'NaN count: {df.isnull().sum().sum()}')"
  ```
  - Expected: 0 (all NaN filled)

- [ ] Check feature columns exist
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); print([c for c in df.columns if 'rms' in c])"
  ```
  - Expected: ch1_rms, ch2_rms, ..., ch8_rms

### Test 2: CWRU Dataset (Fault Classification)

- [ ] Run feature extraction
  ```bash
  python scripts/make_features.py --config configs/cwru.yaml
  ```

- [ ] Verify outputs created
  ```bash
  ls -la data/features/cwru/cwru_features.parquet
  ls -la data/features/cwru/cwru_features.csv
  ls -la data/features/cwru/feature_extraction_log.json
  ```

- [ ] Check fault_type preserved
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/cwru/cwru_features.parquet'); print(df['fault_type'].value_counts())"
  ```
  - Expected: normal, ball_fault, inner_race_fault, outer_race_fault

- [ ] Verify vibration features
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/cwru/cwru_features.parquet'); print([c for c in df.columns if 'vibration_' in c])"
  ```
  - Expected: vibration_mean, vibration_std, vibration_rms, vibration_kurtosis, etc.

### Test 3: AI4I Dataset (Tabular, No Windowing)

- [ ] Run feature extraction
  ```bash
  python scripts/make_features.py --config configs/ai4i.yaml
  ```

- [ ] Verify outputs created
  ```bash
  ls -la data/features/ai4i/ai4i_features.parquet
  ls -la data/features/ai4i/ai4i_features.csv
  ls -la data/features/ai4i/feature_extraction_log.json
  ```

- [ ] Check row count unchanged (no windowing)
  ```bash
  python -c "import pandas as pd; clean = pd.read_parquet('data/clean/ai4i/ai4i_clean.parquet'); feat = pd.read_parquet('data/features/ai4i/ai4i_features.parquet'); print(f'Clean: {len(clean)}, Features: {len(feat)}'); assert len(clean) == len(feat), 'Row count mismatch!'"
  ```
  - Expected: Both should be 10,000 rows

- [ ] Verify derived features added
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ai4i/ai4i_features.parquet'); print('temp_diff' in df.columns, 'power' in df.columns)"
  ```
  - Expected: True True

### Test 4: C-MAPSS Dataset (Turbofan Degradation)

- [ ] Run feature extraction
  ```bash
  python scripts/make_features.py --config configs/fd001.yaml
  ```

- [ ] Verify outputs created
  ```bash
  ls -la data/features/cmapss/fd001_features.parquet
  ls -la data/features/cmapss/fd001_features.csv
  ls -la data/features/cmapss/feature_extraction_log.json
  ```

- [ ] Check sensor features
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/cmapss/fd001_features.parquet'); print([c for c in df.columns if c.startswith('s') and '_' in c][:10])"
  ```
  - Expected: s1_mean, s1_std, s1_rms, s2_mean, etc.

- [ ] Verify unit column preserved
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/cmapss/fd001_features.parquet'); print(f'Units: {df[\"unit\"].nunique()}')"
  ```
  - Expected: 100 units

## Configuration Tests

### Test 5: Custom Feature Subset

- [ ] Edit config to use only 3 features
  ```yaml
  # configs/ims.yaml
  schema:
    computed_features:
      - rms
      - kurtosis
      - mean
  ```

- [ ] Run feature extraction
  ```bash
  python scripts/make_features.py --config configs/ims.yaml
  ```

- [ ] Verify only requested features
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); feature_cols = [c for c in df.columns if any(f in c for f in ['rms', 'kurtosis', 'mean'])]; print(f'Feature columns: {len(feature_cols)}'); assert len(feature_cols) == 8*3, 'Expected 24 feature columns (8 channels × 3 features)'"
  ```

- [ ] Restore original config
  ```bash
  git checkout configs/ims.yaml
  ```

### Test 6: Window Size Modification

- [ ] Edit window parameters
  ```yaml
  # configs/ims.yaml
  prep:
    window:
      size: 1024      # Smaller window
      stride: 512     # Smaller stride
  ```

- [ ] Run feature extraction
  ```bash
  python scripts/make_features.py --config configs/ims.yaml
  ```

- [ ] Verify more windows created (smaller size = more windows)
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); print(f'Rows: {len(df)}')"
  ```
  - Expected: More rows than default (window=2048, stride=1024)

- [ ] Restore original config
  ```bash
  git checkout configs/ims.yaml
  ```

## Edge Case Tests

### Test 7: Small Dataset (max_files)

- [ ] Limit to 10 files
  ```yaml
  # configs/ims.yaml
  paths:
    max_files: 10
  ```

- [ ] Run feature extraction
  ```bash
  python scripts/make_features.py --config configs/ims.yaml
  ```

- [ ] Verify reduced output size
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); print(f'Rows: {len(df)}'); assert len(df) < 2000, 'Expected fewer rows with max_files=10'"
  ```

- [ ] Restore original config
  ```bash
  git checkout configs/ims.yaml
  ```

### Test 8: Error Handling (Missing Input)

- [ ] Test with non-existent config
  ```bash
  python scripts/make_features.py --config configs/nonexistent.yaml
  ```
  - Expected: FileNotFoundError with clear message

- [ ] Test with missing clean data
  ```bash
  # Temporarily rename clean file
  mv data/clean/ims/ims_clean.parquet data/clean/ims/ims_clean.parquet.bak
  python scripts/make_features.py --config configs/ims.yaml
  # Should fail with FileNotFoundError
  mv data/clean/ims/ims_clean.parquet.bak data/clean/ims/ims_clean.parquet
  ```
  - Expected: FileNotFoundError: "Clean data not found"

## Performance Tests

### Test 9: Execution Time

- [ ] Time IMS extraction
  ```bash
  time python scripts/make_features.py --config configs/ims.yaml
  ```
  - Expected: < 2 minutes for 96 files (on 2 vCPU laptop)

- [ ] Time CWRU extraction
  ```bash
  time python scripts/make_features.py --config configs/cwru.yaml
  ```
  - Expected: < 1 minute

### Test 10: Memory Usage

- [ ] Monitor memory during execution
  ```bash
  # On Linux/WSL
  /usr/bin/time -v python scripts/make_features.py --config configs/ims.yaml 2>&1 | grep "Maximum resident"
  ```
  - Expected: < 1 GB peak memory

## Unit Tests

### Test 11: Run Unit Tests

- [ ] Execute test suite
  ```bash
  pytest tests/test_make_features.py -v
  ```
  - Expected: All tests pass

- [ ] Check test coverage
  ```bash
  pytest tests/test_make_features.py --cov=scripts.make_features
  ```
  - Expected: > 70% coverage

## Inspection Tests

### Test 12: Demo Script

- [ ] Run demo for each dataset
  ```bash
  python examples/feature_extraction_demo.py ims
  python examples/feature_extraction_demo.py cwru
  python examples/feature_extraction_demo.py ai4i
  python examples/feature_extraction_demo.py fd001
  ```
  - Expected: No errors, statistics displayed

### Test 13: Manual Inspection

- [ ] Load features in Python
  ```python
  import pandas as pd

  df = pd.read_parquet('data/features/ims/ims_features.parquet')

  # Check shape
  print(f"Shape: {df.shape}")

  # Check columns
  print(f"Columns: {df.columns.tolist()}")

  # Check dtypes
  print(df.dtypes)

  # Check for NaN
  print(f"NaN count: {df.isnull().sum().sum()}")

  # Sample data
  print(df.head())

  # Feature statistics
  feature_cols = [c for c in df.columns if 'rms' in c]
  print(df[feature_cols].describe())
  ```

## Data Quality Tests

### Test 14: Feature Sanity Checks

- [ ] Verify RMS is non-negative
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); rms_cols = [c for c in df.columns if 'rms' in c]; assert (df[rms_cols] >= 0).all().all(), 'RMS should be non-negative'"
  ```

- [ ] Verify peak_to_peak is non-negative
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); ptp_cols = [c for c in df.columns if 'peak_to_peak' in c]; assert (df[ptp_cols] >= 0).all().all(), 'Peak-to-peak should be non-negative'"
  ```

- [ ] Verify std is non-negative
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); std_cols = [c for c in df.columns if '_std' in c]; assert (df[std_cols] >= 0).all().all(), 'Std should be non-negative'"
  ```

### Test 15: Metadata Preservation

- [ ] Check IMS metadata
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ims/ims_features.parquet'); assert 'timestamp' in df.columns; assert 'file_index' in df.columns; assert 'source_file' in df.columns; print('IMS metadata OK')"
  ```

- [ ] Check CWRU metadata
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/cwru/cwru_features.parquet'); assert 'fault_type' in df.columns; assert 'source_file' in df.columns; print('CWRU metadata OK')"
  ```

- [ ] Check AI4I target
  ```bash
  python -c "import pandas as pd; df = pd.read_parquet('data/features/ai4i/ai4i_features.parquet'); assert 'target' in df.columns; print('AI4I target OK')"
  ```

## Output Format Tests

### Test 16: Parquet vs CSV Consistency

- [ ] Compare Parquet and CSV row counts
  ```bash
  python -c "import pandas as pd; pq = pd.read_parquet('data/features/ims/ims_features.parquet'); csv = pd.read_csv('data/features/ims/ims_features.csv'); assert len(pq) == len(csv), 'Row count mismatch'; print(f'Both have {len(pq)} rows')"
  ```

- [ ] Compare Parquet and CSV column counts
  ```bash
  python -c "import pandas as pd; pq = pd.read_parquet('data/features/ims/ims_features.parquet'); csv = pd.read_csv('data/features/ims/ims_features.csv'); assert len(pq.columns) == len(csv.columns), 'Column count mismatch'; print(f'Both have {len(pq.columns)} columns')"
  ```

### Test 17: Metadata Log Validation

- [ ] Check log file exists and is valid JSON
  ```bash
  python -c "import json; log = json.load(open('data/features/ims/feature_extraction_log.json')); print(f'Log valid: {log[\"dataset\"]}')"
  ```

- [ ] Verify log contains expected keys
  ```bash
  python -c "import json; log = json.load(open('data/features/ims/feature_extraction_log.json')); assert 'timestamp' in log; assert 'dataset' in log; assert 'features_computed' in log; assert 'statistics' in log; print('Log structure OK')"
  ```

## Full Pipeline Test

### Test 18: End-to-End Run

- [ ] Clean previous outputs
  ```bash
  rm -rf data/features/*
  ```

- [ ] Run Stage 1 (prep)
  ```bash
  python scripts/prep_data.py --config configs/ims.yaml
  ```

- [ ] Run Stage 2 (features)
  ```bash
  python scripts/make_features.py --config configs/ims.yaml
  ```

- [ ] Verify pipeline completed successfully
  ```bash
  ls -la data/features/ims/
  python examples/feature_extraction_demo.py ims
  ```

## Checklist Summary

Total Tests: 18
- [ ] All basic functionality tests passed (Tests 1-4)
- [ ] All configuration tests passed (Tests 5-6)
- [ ] All edge case tests passed (Tests 7-8)
- [ ] All performance tests passed (Tests 9-10)
- [ ] All unit tests passed (Test 11)
- [ ] All inspection tests passed (Tests 12-13)
- [ ] All data quality tests passed (Tests 14-15)
- [ ] All output format tests passed (Tests 16-17)
- [ ] Full pipeline test passed (Test 18)

## Notes

- If any test fails, check logs for error messages
- Common issues: missing dependencies, incorrect paths, config errors
- For help, see: USAGE_MAKE_FEATURES.md, docs/FEATURE_EXTRACTION.md

## Ready for Stage 3?

Once all tests pass, you're ready to proceed to model training:

```bash
python scripts/train.py --config configs/models/isolation_forest.yaml
```
