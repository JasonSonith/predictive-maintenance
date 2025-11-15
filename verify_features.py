#!/usr/bin/env python3
"""
Verification script for feature datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path


def verify_features(name, parquet_path):
    """Verify a feature dataset"""
    print("=" * 80)
    print(f"VERIFYING: {name}")
    print("=" * 80)

    if not Path(parquet_path).exists():
        print(f"❌ File not found: {parquet_path}")
        return

    # Load data
    df = pd.read_parquet(parquet_path)

    # Basic info
    print(f"\n✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"✓ File size: {Path(parquet_path).stat().st_size / 1024 / 1024:.2f} MB")

    # Columns
    print(f"\n✓ Columns ({len(df.columns)}):")

    # Identify feature columns
    feature_cols = [col for col in df.columns if any(
        feat in col for feat in ['mean', 'std', 'rms', 'min', 'max',
                                  'peak_to_peak', 'kurtosis', 'skewness',
                                  'crest_factor']
    )]

    metadata_cols = [col for col in df.columns if col not in feature_cols]

    print(f"  Metadata columns ({len(metadata_cols)}): {metadata_cols}")
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠ Missing values found:")
        print(missing[missing > 0])
    else:
        print(f"\n✓ No missing values")

    # Check for NaN/inf in features
    if feature_cols:
        has_inf = df[feature_cols].isin([np.inf, -np.inf]).sum().sum()
        if has_inf > 0:
            print(f"⚠ Warning: {has_inf} infinite values in features")
        else:
            print(f"✓ No infinite values")

    # Data types
    print(f"\n✓ Data types:")
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        print(f"  {dtype}: {len(cols)} columns")

    # Feature statistics
    if feature_cols:
        print(f"\n✓ Feature statistics (all computed features):")
        print(f"  Min: {df[feature_cols].min().min():.6f}")
        print(f"  Max: {df[feature_cols].max().max():.6f}")
        print(f"  Mean: {df[feature_cols].mean().mean():.6f}")
        print(f"  Std: {df[feature_cols].std().mean():.6f}")

        # Check for constant features
        constant = [col for col in feature_cols if df[col].nunique() <= 1]
        if constant:
            print(f"  ⚠ Constant features (no variation): {len(constant)} columns")
            print(f"    {constant[:5]}{'...' if len(constant) > 5 else ''}")
        else:
            print(f"  ✓ All features have variation")

    # Dataset-specific checks
    if 'timestamp' in df.columns:
        print(f"\n✓ Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    if 'window_id' in df.columns:
        print(f"✓ Windows: {df['window_id'].nunique()} unique windows")

    if 'sensor' in df.columns:
        print(f"✓ Sensors: {df['sensor'].value_counts().to_dict()}")

    if 'fault_type' in df.columns:
        print(f"✓ Fault types: {df['fault_type'].value_counts().to_dict()}")

    if 'unit' in df.columns:
        print(f"✓ Units: {df['unit'].nunique()} unique units")

    if 'split' in df.columns:
        print(f"✓ Data splits: {df['split'].value_counts().to_dict()}")

    # Sample data
    print(f"\n✓ First 3 rows (showing first 10 columns):")
    print(df.head(3).iloc[:, :10].to_string())

    # Check feature extraction quality
    print(f"\n✓ Feature Extraction Quality Checks:")

    # Check for features that are all zeros
    if feature_cols:
        zero_features = [col for col in feature_cols if (df[col] == 0).all()]
        if zero_features:
            print(f"  ⚠ Features that are all zeros: {len(zero_features)}")
            print(f"    {zero_features[:5]}{'...' if len(zero_features) > 5 else ''}")
        else:
            print(f"  ✓ No features are all zeros")

    # Check for reasonable ranges
    if feature_cols:
        # Check RMS features (should be positive)
        rms_cols = [col for col in feature_cols if 'rms' in col.lower()]
        if rms_cols:
            negative_rms = [(col, df[col].min()) for col in rms_cols if df[col].min() < 0]
            if negative_rms:
                print(f"  ⚠ RMS features with negative values: {negative_rms}")
            else:
                print(f"  ✓ All RMS features are non-negative")

        # Check std features (should be non-negative)
        std_cols = [col for col in feature_cols if 'std' in col.lower()]
        if std_cols:
            negative_std = [(col, df[col].min()) for col in std_cols if df[col].min() < 0]
            if negative_std:
                print(f"  ⚠ STD features with negative values: {negative_std}")
            else:
                print(f"  ✓ All STD features are non-negative")

    print("\n" + "=" * 80)
    print(f"✓ {name} verification complete")
    print("=" * 80 + "\n")


def main():
    datasets = [
        ("IMS Features", "data/features/ims/ims_features.parquet"),
        ("CWRU Features", "data/features/cwru/cwru_features.parquet"),
        ("AI4I Features", "data/features/ai4i/ai4i_features.parquet"),
        ("C-MAPSS FD001 Features", "data/features/cmapss/fd001_features.parquet"),
    ]

    for name, path in datasets:
        try:
            verify_features(name, path)
        except Exception as e:
            print(f"❌ Error verifying {name}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
