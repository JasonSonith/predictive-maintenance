#!/usr/bin/env python3
"""
Threshold Calibration - Stage 4

Calibrates anomaly detection thresholds to achieve a target false-alarm rate (FAR).

This is post-processing: the model outputs continuous anomaly scores, and we need
to find the optimal threshold to convert those scores to binary alerts.

Calibration Strategy:
- Load validation data (assumed mostly normal operation)
- Compute anomaly scores using trained model
- Set threshold at percentile that achieves target FAR
- Save threshold configuration for use in evaluation and scoring

Target FAR examples:
- 0.1/week = ~0.014/day = 1 false alarm every 10 weeks
- 0.05/week = 1 false alarm every 20 weeks (more conservative)
- 0.2/week = 1 false alarm every 5 weeks (more sensitive)
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore', category=FutureWarning)

# PyTorch imports (optional - only for AutoEncoder)
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None


class AutoEncoder:
    """
    PyTorch AutoEncoder wrapper for loading saved models.
    Must match the architecture in train.py.
    """
    def __init__(self, input_dim, encoder_dims, bottleneck_dim, activation='relu',
                 dropout=0.2, device='cpu'):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for AutoEncoder")

        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation
        self.dropout = dropout
        self.device = device
        self.threshold = None

        # Build model
        self.model = self._build_model()
        self.model.to(device)
        self.model.eval()

    def _build_model(self):
        """Build the encoder-decoder network"""
        layers = []

        # Encoder
        prev_dim = self.input_dim
        for hidden_dim in self.encoder_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        # Bottleneck
        layers.append(nn.Linear(prev_dim, self.bottleneck_dim))
        layers.append(self._get_activation())

        # Decoder (mirror of encoder)
        decoder_dims = list(reversed(self.encoder_dims))
        prev_dim = self.bottleneck_dim
        for hidden_dim in decoder_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.input_dim))

        return nn.Sequential(*layers)

    def _get_activation(self):
        """Get activation function"""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _compute_reconstruction_errors(self, X):
        """Compute reconstruction error for each sample"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

        return errors.cpu().numpy()

    def score_samples(self, X):
        """
        Compute anomaly scores (negative reconstruction errors).
        Lower (more negative) scores = more anomalous.
        """
        errors = self._compute_reconstruction_errors(X)
        return -errors


def parse_target_far(far_str):
    """
    Parse target false-alarm rate string.

    Examples:
        "0.1/week" -> 0.1 alarms per week
        "0.05/week" -> 0.05 alarms per week
        "1/month" -> 1 alarm per month (converted to per-week)
        "0.01/day" -> 0.01 alarms per day (converted to per-week)

    Returns:
        float: False alarms per week
    """
    far_str = far_str.lower().strip()

    # Parse value and unit
    if '/week' in far_str:
        value = float(far_str.replace('/week', ''))
        return value
    elif '/month' in far_str:
        value = float(far_str.replace('/month', ''))
        return value / 4.0  # ~4 weeks per month
    elif '/day' in far_str:
        value = float(far_str.replace('/day', ''))
        return value * 7.0  # 7 days per week
    else:
        raise ValueError(f"Invalid FAR format: {far_str}. Use format like '0.1/week', '1/month', or '0.01/day'")


def load_model(model_path, scaler_path, model_type='auto'):
    """
    Load trained model and scaler.

    Args:
        model_path: Path to model file (.joblib or .pth)
        scaler_path: Path to scaler file (.joblib)
        model_type: 'auto', 'sklearn', or 'autoencoder'

    Returns:
        tuple: (model, scaler)
    """
    print(f"\nLoading model: {model_path}")

    # Auto-detect model type from extension
    if model_type == 'auto':
        if str(model_path).endswith('.pth'):
            model_type = 'autoencoder'
        else:
            model_type = 'sklearn'

    # Load model
    if model_type == 'autoencoder':
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for AutoEncoder models")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Recreate model
        model = AutoEncoder(
            input_dim=checkpoint['input_dim'],
            encoder_dims=checkpoint['encoder_dims'],
            bottleneck_dim=checkpoint['bottleneck_dim'],
            activation=checkpoint['activation'],
            dropout=checkpoint['dropout'],
            device='cpu'
        )

        # Load weights
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.threshold = checkpoint['threshold']

        print(f"Loaded AutoEncoder (input_dim={checkpoint['input_dim']})")
    else:
        # Load sklearn model
        model = joblib.load(model_path)
        print(f"Loaded {type(model).__name__}")

    # Load scaler
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        print(f"Loading scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)

    return model, scaler


def load_features(features_path, split='val'):
    """
    Load feature data for a specific split.

    Args:
        features_path: Path to features parquet file
        split: 'train', 'val', or 'test'

    Returns:
        pd.DataFrame: Feature data for the split
    """
    print(f"\nLoading features: {features_path}")
    df = pd.read_parquet(features_path)

    # Filter to specific split if column exists
    if 'split' in df.columns:
        df_split = df[df['split'] == split].copy()
        print(f"Filtered to split='{split}': {len(df_split)} samples")

        if len(df_split) == 0:
            raise ValueError(f"No samples found for split='{split}'. Available splits: {df['split'].unique()}")

        return df_split
    else:
        print(f"No 'split' column found, using all data: {len(df)} samples")
        return df


def select_features(df, feature_names):
    """
    Select feature columns from dataframe.

    Args:
        df: Input dataframe
        feature_names: List of feature names

    Returns:
        np.ndarray: Feature matrix
    """
    # Validate all features exist
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}...")

    X = df[feature_names].values

    # Handle NaN values
    if np.isnan(X).any():
        print("Warning: Found NaN values, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    return X


def compute_anomaly_scores(model, X, model_type='auto'):
    """
    Compute anomaly scores using trained model.

    Args:
        model: Trained anomaly detection model
        X: Feature matrix
        model_type: Model type (used to determine scoring method)

    Returns:
        np.ndarray: Anomaly scores (lower = more anomalous for sklearn convention)
    """
    print("\nComputing anomaly scores...")

    # Determine model type if auto
    if model_type == 'auto':
        model_class_name = type(model).__name__
        if 'SVM' in model_class_name:
            model_type = 'one_class_svm'
        elif model_class_name == 'AutoEncoder':
            model_type = 'autoencoder'
        else:
            model_type = 'default'

    # Get scores based on model type
    if model_type == 'one_class_svm':
        scores = model.decision_function(X)
    else:
        scores = model.score_samples(X)

    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Score mean: {scores.mean():.4f}, std: {scores.std():.4f}")

    return scores


def calibrate_threshold(scores, target_far_per_week, samples_per_week=None):
    """
    Calibrate threshold to achieve target false-alarm rate.

    Strategy:
    1. Assume validation data is mostly normal operation
    2. Target FAR = desired false alarms per week
    3. Set threshold at percentile such that FAR matches target

    Args:
        scores: Anomaly scores from validation data (lower = more anomalous)
        target_far_per_week: Target false alarms per week (e.g., 0.1)
        samples_per_week: Number of samples per week (if known)

    Returns:
        dict: Threshold configuration
    """
    print("\n" + "="*60)
    print("THRESHOLD CALIBRATION")
    print("="*60)

    n_samples = len(scores)
    print(f"\nValidation samples: {n_samples}")
    print(f"Target FAR: {target_far_per_week:.3f} false alarms per week")

    # If we don't know samples_per_week, estimate conservatively
    if samples_per_week is None:
        # Assume ~1 sample per hour as a reasonable default (24*7 = 168 per week)
        # This is conservative - if actual rate is higher, we'll be more sensitive
        samples_per_week = 168
        print(f"Using default sampling rate: {samples_per_week} samples/week")
    else:
        print(f"Sampling rate: {samples_per_week} samples/week")

    # Calculate target false-alarm rate as fraction of samples
    # FAR_percent = (target_far_per_week / samples_per_week) * 100
    target_far_fraction = target_far_per_week / samples_per_week

    print(f"Target FAR as fraction: {target_far_fraction:.6f} ({target_far_fraction*100:.4f}%)")

    # Set threshold at percentile that gives target FAR
    # Since lower scores = anomalies, we use the lower percentile
    threshold_percentile = target_far_fraction * 100
    threshold = np.percentile(scores, threshold_percentile)

    print(f"\nThreshold set at {threshold_percentile:.4f}th percentile: {threshold:.6f}")

    # Validate: count how many samples would trigger alerts
    n_alerts = np.sum(scores < threshold)
    alert_rate = n_alerts / n_samples
    estimated_far_per_week = alert_rate * samples_per_week

    print(f"\nValidation check:")
    print(f"  Samples below threshold: {n_alerts} / {n_samples} ({alert_rate*100:.4f}%)")
    print(f"  Estimated FAR: {estimated_far_per_week:.3f} alarms/week")
    print(f"  Target FAR: {target_far_per_week:.3f} alarms/week")

    if abs(estimated_far_per_week - target_far_per_week) > 0.5:
        print("\nWarning: Large difference between estimated and target FAR")
        print("This may indicate insufficient validation data or unusual score distribution")

    return {
        'threshold': float(threshold),
        'threshold_percentile': float(threshold_percentile),
        'target_far_per_week': float(target_far_per_week),
        'estimated_far_per_week': float(estimated_far_per_week),
        'samples_per_week': int(samples_per_week),
        'n_validation_samples': int(n_samples),
        'n_validation_alerts': int(n_alerts),
        'validation_alert_rate': float(alert_rate),
        'score_min': float(scores.min()),
        'score_max': float(scores.max()),
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std())
    }


def main():
    """
    Main entry point for threshold calibration.

    Usage examples:
        # Basic usage with model config
        python scripts/threshold.py --config configs/models/isolation_forest.yaml --target_far 0.1/week

        # Specify model and scaler directly
        python scripts/threshold.py \
            --model artifacts/models/ims_iforest.joblib \
            --scaler artifacts/models/ims_iforest_scaler.joblib \
            --features data/features/ims/ims_features.parquet \
            --target_far 0.1/week \
            --output artifacts/reports/ims_iforest/

        # Use test data instead of validation
        python scripts/threshold.py --config configs/models/isolation_forest.yaml \
            --target_far 0.1/week --split test

        # Specify sampling rate explicitly
        python scripts/threshold.py --config configs/models/isolation_forest.yaml \
            --target_far 0.1/week --samples_per_week 672
    """
    parser = argparse.ArgumentParser(
        description='Calibrate anomaly detection threshold to target false-alarm rate',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options
    parser.add_argument('--config', help='Model config YAML file')
    parser.add_argument('--model', help='Path to trained model (.joblib or .pth)')
    parser.add_argument('--scaler', help='Path to scaler (.joblib)')
    parser.add_argument('--features', help='Path to features parquet file')

    # Threshold settings
    parser.add_argument('--target_far', required=True,
                       help='Target false-alarm rate (e.g., "0.1/week", "1/month", "0.01/day")')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                       help='Data split to use for calibration (default: val)')
    parser.add_argument('--samples_per_week', type=int,
                       help='Number of samples per week (if known, improves accuracy)')

    # Output
    parser.add_argument('--output', help='Output directory for threshold config')

    args = parser.parse_args()

    print("="*60)
    print("THRESHOLD CALIBRATION - STAGE 4")
    print("="*60)

    # Parse target FAR
    target_far_per_week = parse_target_far(args.target_far)

    # Load configuration if provided
    if args.config:
        print(f"\nLoading model config: {args.config}")
        with open(args.config, 'r') as f:
            model_config = yaml.safe_load(f)

        # Load dataset config
        dataset_config_path = model_config['dataset_config']
        print(f"Loading dataset config: {dataset_config_path}")
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        # Get paths from config
        model_path = Path(model_config['paths']['model_output'])
        scaler_path = Path(model_config['paths']['scaler_output'])
        features_path = Path(dataset_config['paths']['features_output_path'])
        output_dir = Path(model_config['paths']['report_dir'])

        # Load feature list
        features_file = output_dir / 'features.txt'
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        with open(features_file, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]

    else:
        # Use command-line paths
        if not all([args.model, args.features]):
            parser.error("Either --config or both --model and --features are required")

        model_path = Path(args.model)
        scaler_path = Path(args.scaler) if args.scaler else None
        features_path = Path(args.features)
        output_dir = Path(args.output) if args.output else model_path.parent.parent / 'reports' / model_path.stem

        # Feature names will need to be inferred
        feature_names = None

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and scaler
    model, scaler = load_model(model_path, scaler_path)

    # Load features
    df = load_features(features_path, split=args.split)

    # Select features
    if feature_names:
        X = select_features(df, feature_names)
    else:
        # Use all numeric columns (excluding metadata)
        metadata_cols = ['timestamp', 'file_index', 'source_file', 'window_id',
                        'sensor', 'split', 'unit', 'cycle', 'fault_type']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        X = df[feature_cols].values
        feature_names = feature_cols

    print(f"\nFeature matrix shape: {X.shape}")

    # Apply scaling
    if scaler is not None:
        print("Applying feature scaling...")
        X = scaler.transform(X)

    # Compute anomaly scores
    scores = compute_anomaly_scores(model, X)

    # Calibrate threshold
    threshold_config = calibrate_threshold(
        scores,
        target_far_per_week,
        samples_per_week=args.samples_per_week
    )

    # Add metadata
    threshold_config['timestamp'] = datetime.now().isoformat()
    threshold_config['model_path'] = str(model_path)
    threshold_config['features_path'] = str(features_path)
    threshold_config['calibration_split'] = args.split
    threshold_config['n_features'] = X.shape[1]

    # Save threshold configuration
    threshold_path = output_dir / 'threshold_config.json'
    print(f"\nSaving threshold config: {threshold_path}")
    with open(threshold_path, 'w') as f:
        json.dump(threshold_config, f, indent=2)

    # Also save a simple threshold file (just the value)
    threshold_value_path = output_dir / 'threshold.txt'
    with open(threshold_value_path, 'w') as f:
        f.write(f"{threshold_config['threshold']:.6f}\n")

    print("\n" + "="*60)
    print("THRESHOLD CALIBRATION COMPLETE")
    print("="*60)
    print(f"\nThreshold: {threshold_config['threshold']:.6f}")
    print(f"Configuration saved to: {threshold_path}")
    print(f"  python scripts/evaluate.py --report {output_dir}")


if __name__ == '__main__':
    main()
