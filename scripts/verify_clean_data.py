#!/usr/bin/env python3
"""
Quick verification script for cleaned datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path


def verify_dataset(name, parquet_path):
    """Verify a cleaned dataset"""
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
    print(f"  {list(df.columns)}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠ Missing values found:")
        print(missing[missing > 0])
    else:
        print(f"\n✓ No missing values")

    # Data types
    print(f"\n✓ Data types:")
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        print(f"  {dtype}: {len(cols)} columns")

    # Dataset-specific checks
    if 'timestamp' in df.columns:
        print(f"\n✓ Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        if df['timestamp'].dtype != 'datetime64[ns]':
            print(f"  ⚠ Warning: timestamp is {df['timestamp'].dtype}, not datetime64")

    if 'file_index' in df.columns:
        print(f"✓ Unique files: {df['file_index'].nunique()}")

    if 'unit' in df.columns:
        print(f"✓ Unique units: {df['unit'].nunique()}")

    if 'fault_type' in df.columns:
        print(f"✓ Fault types: {df['fault_type'].value_counts().to_dict()}")

    # Sensor columns
    sensor_cols = [col for col in df.columns if col.startswith('ch') or col.startswith('s') and col[1:].isdigit()]
    if sensor_cols:
        print(f"\n✓ Sensor columns ({len(sensor_cols)}): {sensor_cols[:5]}{'...' if len(sensor_cols) > 5 else ''}")
        print(f"  Value range: [{df[sensor_cols].min().min():.4f}, {df[sensor_cols].max().max():.4f}]")

        # Check for constant columns
        constant = [col for col in sensor_cols if df[col].nunique() <= 1]
        if constant:
            print(f"  ⚠ Constant columns (no variation): {constant}")

    # Vibration amplitude (CWRU)
    if 'vibration_amp' in df.columns:
        print(f"\n✓ Vibration amplitude range: [{df['vibration_amp'].min():.4f}, {df['vibration_amp'].max():.4f}]")

    # Sample data
    print(f"\n✓ First 3 rows:")
    print(df.head(3).to_string())

    # Summary stats
    print(f"\n✓ Summary statistics (numeric columns):")
    print(df.describe().to_string())

    print("\n" + "=" * 80)
    print(f"✓ {name} verification complete")
    print("=" * 80 + "\n")


def main():
    datasets = [
        ("IMS", "data/clean/ims/ims_clean.parquet"),
        ("CWRU", "data/clean/cwru/cwru_clean.parquet"),
        ("AI4I", "data/clean/ai4i/ai4i_clean.parquet"),
        ("C-MAPSS FD001", "data/clean/cmapss/fd001_clean.parquet"),
    ]

    for name, path in datasets:
        try:
            verify_dataset(name, path)
        except Exception as e:
            print(f"❌ Error verifying {name}: {e}\n")


if __name__ == '__main__':
    main()
