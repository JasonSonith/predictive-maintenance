#!/usr/bin/env python3
"""
Generate Top Features Visualization Only
Creates a styled bar chart showing the top 10 most important features.
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Generate top features visualization')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.joblib)')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to feature data (CSV or Parquet)')
    parser.add_argument('--output', type=str, default='top_features.png',
                        help='Output file path (default: top_features.png)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top features to show (default: 10)')
    parser.add_argument('--sample-size', type=int, default=200,
                        help='Number of samples for SHAP (default: 200)')
    return parser.parse_args()

def extract_features(df):
    """Extract numeric feature columns (remove metadata)"""
    metadata_cols = ['timestamp', 'file_id', 'unit_id', 'anomaly', 'label',
                     'fault_type', 'cycle', 'rul', 'unit', 'file_index',
                     'source_file', 'window_id', 'sensor', 'split']

    feature_cols = [col for col in df.columns if col not in metadata_cols]
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])

    return numeric_df

def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = joblib.load(args.model)

    print(f"Loading features: {args.features}")
    if args.features.endswith('.parquet'):
        df = pd.read_parquet(args.features)
    else:
        df = pd.read_csv(args.features)

    X_features = extract_features(df)
    print(f"Extracted {len(X_features.columns)} features from {len(X_features)} samples")

    # Sample for SHAP performance
    if len(X_features) > args.sample_size:
        print(f"Sampling {args.sample_size} rows for SHAP computation...")
        X_sample = X_features.sample(n=args.sample_size, random_state=42)
    else:
        X_sample = X_features

    # Compute SHAP values
    print("Computing SHAP values (this may take a minute)...")
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    print("✅ SHAP computation complete")

    # Calculate feature importance
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = X_sample.columns.values
    top_indices = np.argsort(feature_importance)[-args.top_n:]

    # Create visualization
    print(f"Generating top {args.top_n} features plot...")
    plt.figure(figsize=(12, 7))

    # Use viridis colormap (yellow -> green -> blue gradient)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, args.top_n))

    plt.barh(range(args.top_n), feature_importance[top_indices], color=colors)
    plt.yticks(range(args.top_n), [feature_names[i] for i in top_indices], fontsize=11)
    plt.xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=13)
    plt.title(f'Top {args.top_n} Most Important Features',
              fontsize=15, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}")
    print(f"\nTop 3 features:")
    for i, idx in enumerate(top_indices[-3:][::-1], 1):
        print(f"  {i}. {feature_names[idx]} (importance: {feature_importance[idx]:.4f})")

if __name__ == '__main__':
    main()
