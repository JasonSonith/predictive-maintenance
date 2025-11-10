#!/usr/bin/env python3
"""
Feature Extraction Demo

This script demonstrates how to use make_features.py and inspect the results.

Usage:
    python examples/feature_extraction_demo.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def demo_inspect_features(dataset='ims'):
    """
    Demonstrate how to inspect extracted features.

    Args:
        dataset: 'ims', 'cwru', 'ai4i', or 'fd001'
    """
    print(f"\n{'='*80}")
    print(f"Feature Extraction Demo - {dataset.upper()} Dataset")
    print(f"{'='*80}\n")

    # Construct paths
    features_dir = Path('data/features') / dataset
    parquet_file = features_dir / f'{dataset}_features.parquet'
    csv_file = features_dir / f'{dataset}_features.csv'
    log_file = features_dir / 'feature_extraction_log.json'

    # Check if features exist
    if not parquet_file.exists():
        print(f"Features not found at: {parquet_file}")
        print(f"\nPlease run feature extraction first:")
        print(f"  python scripts/make_features.py --config configs/{dataset}.yaml")
        return

    print(f"Loading features from: {parquet_file}")

    # Load features
    df = pd.read_parquet(parquet_file)

    print(f"\n{'-'*80}")
    print("Dataset Overview")
    print(f"{'-'*80}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Display columns
    print(f"\n{'-'*80}")
    print("Columns")
    print(f"{'-'*80}")

    # Separate metadata and feature columns
    feature_keywords = ['mean', 'std', 'rms', 'kurtosis', 'skewness',
                       'peak_to_peak', 'crest_factor', 'min', 'max',
                       'median', 'temp_diff', 'power']

    feature_cols = [col for col in df.columns if any(kw in col for kw in feature_keywords)]
    metadata_cols = [col for col in df.columns if col not in feature_cols]

    print(f"\nMetadata columns ({len(metadata_cols)}):")
    for col in metadata_cols[:10]:  # Show first 10
        print(f"  - {col}")
    if len(metadata_cols) > 10:
        print(f"  ... and {len(metadata_cols) - 10} more")

    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols[:10]:  # Show first 10
        print(f"  - {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")

    # Show sample data
    print(f"\n{'-'*80}")
    print("Sample Data (first 3 rows)")
    print(f"{'-'*80}")

    # Show subset of columns for readability
    display_cols = metadata_cols[:3] + feature_cols[:5]
    print(df[display_cols].head(3).to_string(index=False))

    # Show statistics
    print(f"\n{'-'*80}")
    print("Feature Statistics")
    print(f"{'-'*80}")

    if feature_cols:
        stats = df[feature_cols[:5]].describe()  # Show first 5 features
        print(stats.to_string())

    # Check for NaN values
    print(f"\n{'-'*80}")
    print("Data Quality")
    print(f"{'-'*80}")

    nan_counts = df.isnull().sum()
    if nan_counts.sum() == 0:
        print("✓ No missing values")
    else:
        print("Missing values found:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")

    # Load and display log if available
    if log_file.exists():
        print(f"\n{'-'*80}")
        print("Extraction Log")
        print(f"{'-'*80}")

        import json
        with open(log_file) as f:
            log = json.load(f)

        print(f"Timestamp: {log.get('timestamp', 'N/A')}")
        print(f"Dataset: {log.get('dataset', 'N/A')}")
        print(f"Features computed: {', '.join(log.get('features_computed', []))}")
        print(f"\nStatistics:")
        for key, value in log.get('statistics', {}).items():
            print(f"  - {key}: {value}")

    print(f"\n{'='*80}")
    print("Demo Complete!")
    print(f"{'='*80}\n")


def demo_visualize_features(dataset='ims'):
    """
    Visualize feature distributions and trends.

    Args:
        dataset: 'ims', 'cwru', 'ai4i', or 'fd001'
    """
    print(f"\nVisualizing features for {dataset.upper()}...")

    features_file = Path('data/features') / dataset / f'{dataset}_features.parquet'

    if not features_file.exists():
        print(f"Features not found. Please run feature extraction first.")
        return

    df = pd.read_parquet(features_file)

    # Find RMS columns (common across all datasets)
    rms_cols = [col for col in df.columns if 'rms' in col.lower()]

    if not rms_cols:
        print("No RMS features found.")
        return

    # Plot RMS over time/windows
    fig, axes = plt.subplots(len(rms_cols[:4]), 1, figsize=(12, 8), sharex=True)

    if len(rms_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(rms_cols[:4]):  # Plot first 4 RMS features
        ax = axes[idx] if len(rms_cols) > 1 else axes[0]

        df[col].plot(ax=ax, alpha=0.7)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

        # Highlight anomalies (values > mean + 3*std)
        threshold = df[col].mean() + 3 * df[col].std()
        ax.axhline(threshold, color='r', linestyle='--', alpha=0.5, label='3σ threshold')
        ax.legend()

    axes[-1].set_xlabel('Window Index')
    plt.suptitle(f'{dataset.upper()} - RMS Features Over Time')
    plt.tight_layout()

    output_dir = Path('artifacts/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset}_rms_features.png'

    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")

    # Don't show plot (for headless environments)
    # plt.show()
    plt.close()


def demo_compare_features():
    """
    Compare feature distributions across different sensors/channels.
    """
    print("\nComparing feature distributions...")

    # Load IMS features as example
    features_file = Path('data/features/ims/ims_features.parquet')

    if not features_file.exists():
        print("IMS features not found.")
        return

    df = pd.read_parquet(features_file)

    # Find all kurtosis features
    kurtosis_cols = [col for col in df.columns if 'kurtosis' in col.lower()]

    if len(kurtosis_cols) < 2:
        print("Not enough kurtosis features to compare.")
        return

    # Create box plots
    fig, ax = plt.subplots(figsize=(12, 6))

    df[kurtosis_cols[:8]].boxplot(ax=ax)  # First 8 channels
    ax.set_ylabel('Kurtosis')
    ax.set_title('Kurtosis Distribution by Channel')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_dir = Path('artifacts/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'kurtosis_comparison.png'

    plt.savefig(output_file, dpi=150)
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


if __name__ == '__main__':
    import sys

    # Determine which dataset to demo
    dataset = 'ims'
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()

    # Run demos
    demo_inspect_features(dataset)

    # Uncomment to generate visualizations
    # demo_visualize_features(dataset)
    # demo_compare_features()

    print("\nTo run with a different dataset:")
    print("  python examples/feature_extraction_demo.py cwru")
    print("  python examples/feature_extraction_demo.py ai4i")
    print("  python examples/feature_extraction_demo.py fd001")
