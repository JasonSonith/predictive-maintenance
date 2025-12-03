#!/usr/bin/env python3
"""
SHAP Visualization Script
Generates feature importance and anomaly detection visualizations for presentations.
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate SHAP visualizations')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.joblib)')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to feature data (CSV or Parquet)')
    parser.add_argument('--scores', type=str, default=None,
                        help='Path to scored data with anomaly scores (optional)')
    parser.add_argument('--output', type=str, default='artifacts/figures',
                        help='Output directory for plots')
    parser.add_argument('--sample-size', type=int, default=200,
                        help='Number of samples for SHAP (default: 200)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Anomaly threshold for visualization')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    return parser.parse_args()

def load_data(features_path, verbose=False):
    """Load feature data from CSV or Parquet"""
    if verbose:
        print(f"Loading features from: {features_path}")

    if features_path.endswith('.parquet'):
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path)

    if verbose:
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    return df

def extract_features(df, verbose=False):
    """Extract feature columns (remove metadata)"""
    # Common metadata columns to exclude
    metadata_cols = ['timestamp', 'file_id', 'unit_id', 'anomaly', 'label',
                     'fault_type', 'cycle', 'rul', 'unit', 'file_index',
                     'source_file', 'window_id', 'sensor', 'split']

    # First remove known metadata columns
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Then filter to only numeric columns
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = numeric_df.columns.tolist()

    if verbose:
        print(f"Extracted {len(feature_cols)} numeric feature columns")
        print(f"  Features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"  Features: {feature_cols}")

    return numeric_df, feature_cols

def load_threshold(model_path):
    """Try to load calibrated threshold from model artifacts"""
    model_dir = Path(model_path).parent
    threshold_file = model_dir.parent / 'reports' / Path(model_path).stem / 'threshold_config.json'

    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            config = json.load(f)
            return config.get('calibrated_threshold')
    return None

def plot_shap_summary(shap_values, X_sample, output_dir, verbose=False):
    """Generate SHAP summary plot (beeswarm)"""
    if verbose:
        print("Generating SHAP summary plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    output_path = output_dir / 'shap_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  ✅ Saved: {output_path}")

def plot_shap_bar(shap_values, X_sample, output_dir, verbose=False):
    """Generate SHAP bar plot (feature importance)"""
    if verbose:
        print("Generating SHAP bar plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    output_path = output_dir / 'shap_bar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  ✅ Saved: {output_path}")

def plot_top_features(shap_values, feature_names, output_dir, top_n=10, verbose=False):
    """Generate top N features bar chart"""
    if verbose:
        print(f"Generating top {top_n} features plot...")

    # Calculate mean absolute SHAP values
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(feature_importance)[-top_n:]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    plt.barh(range(top_n), feature_importance[top_indices], color=colors)
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'top_features.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  ✅ Saved: {output_path}")
        print(f"  Top 3 features: {[feature_names[i] for i in top_indices[-3:][::-1]]}")

def plot_anomaly_timeline(scores_df, threshold, output_dir, verbose=False):
    """Generate anomaly score timeline"""
    if verbose:
        print("Generating anomaly score timeline...")

    plt.figure(figsize=(14, 6))

    # Plot scores
    plt.plot(range(len(scores_df)), scores_df['anomaly_score'],
             linewidth=1, alpha=0.7, color='#2E86AB', label='Anomaly Score')

    # Add threshold line
    if threshold is not None:
        plt.axhline(y=threshold, color='#A23B72', linestyle='--',
                   linewidth=2, label=f'Threshold ({threshold:.3f})')

    # Highlight anomalies if alert column exists
    if 'alert' in scores_df.columns:
        anomaly_indices = scores_df[scores_df['alert'] == 1].index
        if len(anomaly_indices) > 0:
            plt.scatter(anomaly_indices, scores_df.loc[anomaly_indices, 'anomaly_score'],
                       color='red', s=20, alpha=0.6, label='Detected Anomalies', zorder=5)

    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title('Anomaly Detection Timeline', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'anomaly_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  ✅ Saved: {output_path}")
        if 'alert' in scores_df.columns:
            n_anomalies = scores_df['alert'].sum()
            print(f"  Detected {n_anomalies} anomalies ({n_anomalies/len(scores_df)*100:.2f}%)")

def plot_score_distribution(scores_df, threshold, output_dir, verbose=False):
    """Generate anomaly score distribution histogram"""
    if verbose:
        print("Generating score distribution plot...")

    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.hist(scores_df['anomaly_score'], bins=50, edgecolor='black',
             alpha=0.7, color='#2E86AB')

    # Add threshold line
    if threshold is not None:
        plt.axvline(x=threshold, color='#A23B72', linestyle='--',
                   linewidth=2, label=f'Threshold ({threshold:.3f})')

    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'score_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  ✅ Saved: {output_path}")
        print(f"  Score range: [{scores_df['anomaly_score'].min():.3f}, {scores_df['anomaly_score'].max():.3f}]")
        print(f"  Mean score: {scores_df['anomaly_score'].mean():.3f}")

def plot_feature_correlations(X_features, output_dir, top_n=15, verbose=False):
    """Generate correlation heatmap for top features"""
    if verbose:
        print(f"Generating correlation heatmap (top {top_n} features)...")

    # Select top N features by variance
    feature_variance = X_features.var()
    top_features = feature_variance.nlargest(top_n).index
    X_subset = X_features[top_features]

    # Compute correlation matrix
    corr_matrix = X_subset.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Feature Correlation Heatmap (Top {top_n} Features)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'feature_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  ✅ Saved: {output_path}")

def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("="*60)
        print("SHAP Visualization Generator")
        print("="*60)

    # Load model
    if args.verbose:
        print(f"\n📦 Loading model: {args.model}")
    model = joblib.load(args.model)

    # Load features
    df = load_data(args.features, args.verbose)
    X_features, feature_names = extract_features(df, args.verbose)

    # Sample for SHAP (SHAP is slow on large datasets)
    if len(X_features) > args.sample_size:
        if args.verbose:
            print(f"\n🎲 Sampling {args.sample_size} rows for SHAP analysis...")
        X_sample = X_features.sample(n=args.sample_size, random_state=42)
    else:
        X_sample = X_features
        if args.verbose:
            print(f"\n✓ Using all {len(X_features)} rows for SHAP")

    # Generate SHAP values
    if args.verbose:
        print("\n🧮 Computing SHAP values (this may take a minute)...")

    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        if args.verbose:
            print("  ✅ SHAP computation complete")
    except Exception as e:
        print(f"  ⚠️  Error computing SHAP: {e}")
        print("  Trying TreeExplainer (for tree-based models)...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)
            if args.verbose:
                print("  ✅ SHAP computation complete (TreeExplainer)")
        except Exception as e2:
            print(f"  ❌ SHAP failed: {e2}")
            print("  Skipping SHAP plots...")
            shap_values = None

    # Generate SHAP plots
    if shap_values is not None:
        if args.verbose:
            print("\n📊 Generating SHAP plots...")

        plot_shap_summary(shap_values, X_sample, output_dir, args.verbose)
        plot_shap_bar(shap_values, X_sample, output_dir, args.verbose)
        plot_top_features(shap_values, feature_names, output_dir, verbose=args.verbose)

    # Generate additional plots if scores are provided
    if args.scores:
        if args.verbose:
            print("\n📈 Generating anomaly detection plots...")

        scores_df = load_data(args.scores, args.verbose)

        # Get threshold
        threshold = args.threshold
        if threshold is None:
            threshold = load_threshold(args.model)
            if threshold and args.verbose:
                print(f"  Loaded calibrated threshold: {threshold:.4f}")

        if 'anomaly_score' in scores_df.columns:
            plot_anomaly_timeline(scores_df, threshold, output_dir, args.verbose)
            plot_score_distribution(scores_df, threshold, output_dir, args.verbose)
        else:
            print("  ⚠️  No 'anomaly_score' column found in scores file")

    # Generate correlation heatmap
    if args.verbose:
        print("\n🔥 Generating correlation heatmap...")
    plot_feature_correlations(X_features, output_dir, verbose=args.verbose)

    # Summary
    if args.verbose:
        print("\n" + "="*60)
        print(f"✅ All visualizations saved to: {output_dir}/")
        print("="*60)
        print("\nGenerated files:")
        for file in sorted(output_dir.glob('*.png')):
            print(f"  - {file.name}")

if __name__ == '__main__':
    main()
