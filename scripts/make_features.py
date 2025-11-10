#!/usr/bin/env python3
"""
Feature Engineering Script - Stage 2

Computes time-domain features from rolling windows.
Usage: python scripts/make_features.py --config configs/ims.yaml
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import kurtosis, skew
from tqdm import tqdm

# Suppress scipy precision warnings for nearly constant data
warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_features(data):
    """Compute basic time-domain features from a window of data"""
    if len(data) == 0:
        return {}

    # Remove NaN values
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return {}

    features = {}
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['min'] = np.min(data)
    features['max'] = np.max(data)
    features['rms'] = np.sqrt(np.mean(data**2))
    features['peak_to_peak'] = np.max(data) - np.min(data)

    # Kurtosis and skewness (need enough samples)
    if len(data) >= 4:
        features['kurtosis'] = kurtosis(data, fisher=True)
    else:
        features['kurtosis'] = 0.0

    if len(data) >= 3:
        features['skewness'] = skew(data)
    else:
        features['skewness'] = 0.0

    # Crest factor
    rms = features['rms']
    if rms > 0:
        features['crest_factor'] = np.max(np.abs(data)) / rms
    else:
        features['crest_factor'] = 0.0

    return features


def make_windows(data, window_size, stride):
    """Create sliding windows from data"""
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return windows


def extract_ims_features(df, config):
    """Extract features for IMS dataset (8 channels)"""
    print("Processing IMS dataset...")

    # Get config params
    window_size = config['prep']['window']['size']
    stride = config['prep']['window']['stride']

    # Find sensor columns
    sensor_cols = [col for col in df.columns if col.startswith('ch')]
    print(f"Sensors: {sensor_cols}")
    print(f"Window size: {window_size}, stride: {stride}")

    # Process each file
    results = []
    for file_idx, file_df in tqdm(list(df.groupby('file_index')), desc="Files"):
        file_df = file_df.sort_index()

        # Get metadata from first row
        metadata = {
            'timestamp': file_df['timestamp'].iloc[0],
            'file_index': file_idx,
            'source_file': file_df['source_file'].iloc[0]
        }

        # Process each sensor
        for sensor in sensor_cols:
            signal = file_df[sensor].values
            windows = make_windows(signal, window_size, stride)

            # Compute features for each window
            for win_idx, window in enumerate(windows):
                feats = compute_features(window)

                row = metadata.copy()
                row['window_id'] = len(results)
                row['sensor'] = sensor

                # Add features with sensor prefix
                for feat_name, feat_val in feats.items():
                    row[f'{sensor}_{feat_name}'] = feat_val

                results.append(row)

    print(f"Created {len(results)} feature vectors")
    return pd.DataFrame(results)


def extract_cwru_features(df, config):
    """Extract features for CWRU dataset (vibration data)"""
    print("Processing CWRU dataset...")

    # Get config params
    window_size = config['prep']['window']['size']
    stride = config['prep']['window']['stride']

    print(f"Window size: {window_size}, stride: {stride}")

    # Process each file
    results = []
    for file_name, file_df in tqdm(list(df.groupby('source_file')), desc="Files"):
        if 'time_index' in file_df.columns:
            file_df = file_df.sort_values('time_index')

        # Get metadata
        metadata = {}
        for col in ['source_file', 'fault_type', 'fault_size_mils', 'load_hp', 'sensor_location']:
            if col in file_df.columns:
                metadata[col] = file_df[col].iloc[0]

        # Get vibration signal
        signal = file_df['vibration_amp'].values
        windows = make_windows(signal, window_size, stride)

        # Compute features for each window
        for win_idx, window in enumerate(windows):
            feats = compute_features(window)

            row = metadata.copy()
            row['window_id'] = len(results)

            # Add features with vibration prefix
            for feat_name, feat_val in feats.items():
                row[f'vibration_{feat_name}'] = feat_val

            results.append(row)

    print(f"Created {len(results)} feature vectors")
    return pd.DataFrame(results)


def extract_ai4i_features(df, config):
    """Extract features for AI4I dataset (no windowing needed)"""
    print("Processing AI4I dataset (tabular, no windowing)...")

    # Copy dataframe
    features_df = df.copy()

    # Add derived features
    if 'air_temp_k' in features_df.columns and 'proc_temp_k' in features_df.columns:
        features_df['temp_diff'] = features_df['proc_temp_k'] - features_df['air_temp_k']

    if 'torque_nm' in features_df.columns and 'rpm' in features_df.columns:
        features_df['power'] = features_df['torque_nm'] * features_df['rpm']

    print(f"Features: {len(features_df)} rows, {len(features_df.columns)} columns")
    return features_df


def extract_cmapss_features(df, config):
    """Extract features for C-MAPSS dataset (21 sensors)"""
    print("Processing C-MAPSS dataset...")

    # Get config params
    window_size = config['prep'].get('window_params', {}).get('window_size', 30)
    stride = config['prep'].get('window_params', {}).get('step_size', 1)

    print(f"Window size: {window_size}, stride: {stride}")

    # Find sensor columns (s1, s2, ..., s21)
    sensor_cols = [col for col in df.columns if col.startswith('s') and col[1:].isdigit()]
    print(f"Sensors: {sensor_cols[:5]}... ({len(sensor_cols)} total)")

    # Process each unit (engine)
    results = []
    for unit_id, unit_df in tqdm(list(df.groupby('unit')), desc="Units"):
        if 'cycle' in unit_df.columns:
            unit_df = unit_df.sort_values('cycle')

        # Get the number of windows we'll create
        num_windows = (len(unit_df) - window_size) // stride + 1

        # Create windows for this unit
        for win_idx in range(num_windows):
            start_idx = win_idx * stride
            end_idx = start_idx + window_size

            if end_idx <= len(unit_df):
                # Get metadata from end of window
                row = {}
                for col in ['unit', 'cycle', 'split']:
                    if col in unit_df.columns:
                        row[col] = unit_df.iloc[end_idx - 1][col]

                row['window_id'] = len(results)

                # Compute features for ALL sensors in this window
                for sensor in sensor_cols:
                    signal = unit_df[sensor].values[start_idx:end_idx]
                    feats = compute_features(signal)

                    # Add features with sensor prefix
                    for feat_name, feat_val in feats.items():
                        row[f'{sensor}_{feat_name}'] = feat_val

                results.append(row)

    print(f"Created {len(results)} feature vectors")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Feature Engineering')
    parser.add_argument('--config', required=True, help='Config YAML file')
    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Load cleaned data
    clean_path = Path(config['paths']['clean_output_path'])
    print(f"Loading data: {clean_path}")
    df = pd.read_parquet(clean_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Detect dataset type and extract features
    paths = config.get('paths', {})

    if 'raw_input_dir' in paths:
        features_df = extract_ims_features(df, config)
    elif 'raw_input_dirs' in paths:
        features_df = extract_cwru_features(df, config)
    elif 'raw_input_file' in paths:
        features_df = extract_ai4i_features(df, config)
    elif 'raw_train_file' in paths or 'raw_test_file' in paths:
        features_df = extract_cmapss_features(df, config)
    else:
        raise ValueError("Unknown dataset type")

    # Fill any NaN values
    if features_df.isnull().sum().sum() > 0:
        print("Warning: Found NaN values, filling with 0")
        features_df.fillna(0.0, inplace=True)

    # Save Parquet
    parquet_path = Path(config['paths']['features_output_path'])
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(parquet_path, index=False, compression='snappy')
    print(f"Saved Parquet: {parquet_path}")

    # Save CSV
    csv_path = Path(config['paths']['features_output_path_csv'])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("=" * 60)
    print(f"Done! {len(features_df)} feature vectors, {len(features_df.columns)} columns")
    print("=" * 60)


if __name__ == '__main__':
    main()
