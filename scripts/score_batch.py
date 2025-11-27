#!/usr/bin/env python3
"""
Batch Scoring - Stage 6

Applies trained anomaly detection models to new sensor data for production inference.

This script:
1. Loads a trained model and calibrated threshold
2. Scores input features to generate anomaly predictions
3. Saves predictions with metadata for downstream use

Usage:
    python scripts/score_batch.py \\
        --model artifacts/models/ims_iforest.joblib \\
        --input data/features/ims_test.csv \\
        --output artifacts/scores/ims_test_scores

    python scripts/score_batch.py \\
        --model artifacts/models/cwru_iforest.joblib \\
        --input data/features/cwru_test.csv \\
        --output artifacts/scores/cwru_test_scores \\
        --threshold 0.55 \\
        --verbose
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
            return nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def score_samples(self, X):
        """
        Compute anomaly scores (reconstruction error).
        Returns negative errors to match sklearn convention (lower = more anomalous).
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            X_pred = self.model(X_tensor)
            errors = torch.mean((X_tensor - X_pred) ** 2, dim=1)
            # Return negative errors (sklearn convention: lower = more anomalous)
            return -errors.cpu().numpy()


def load_model(model_path, scaler_path=None, verbose=False):
    """
    Load trained model and optional scaler.

    Args:
        model_path: Path to trained model (.joblib or .pth)
        scaler_path: Path to scaler (.joblib), optional
        verbose: Print detailed loading info

    Returns:
        tuple: (model, scaler, model_type, model_name)
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"ERROR: Model file not found: {model_path}\n\n"
            f"Solution: Check that the model path is correct. List available models with:\n"
            f"  ls artifacts/models/"
        )

    if verbose:
        print(f"\nLoading model: {model_path}")

    # Determine model type from file extension
    if model_path.suffix == '.pth':
        # Load PyTorch AutoEncoder
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for AutoEncoder models. Install with: pip install torch")

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        model = AutoEncoder(
            input_dim=checkpoint['input_dim'],
            encoder_dims=checkpoint['encoder_dims'],
            bottleneck_dim=checkpoint['bottleneck_dim'],
            activation=checkpoint.get('activation', 'relu'),
            dropout=checkpoint.get('dropout', 0.2),
            device='cpu'
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.threshold = checkpoint.get('threshold')
        model_type = 'autoencoder'
        model_name = model_path.stem

        if verbose:
            print(f"  Type: AutoEncoder")
            print(f"  Input dim: {checkpoint['input_dim']}")
            print(f"  Architecture: {checkpoint['encoder_dims']} -> {checkpoint['bottleneck_dim']}")
    else:
        # Load sklearn model
        model = joblib.load(model_path)
        model_name = model_path.stem

        # Detect model type
        model_class = type(model).__name__
        if 'IsolationForest' in model_class:
            model_type = 'isolation_forest'
        elif 'LocalOutlierFactor' in model_class:
            model_type = 'knn_lof'
        elif 'OneClassSVM' in model_class or 'SVM' in model_class:
            model_type = 'one_class_svm'
        else:
            model_type = 'unknown'

        if verbose:
            print(f"  Type: {model_class}")

    # Load scaler if provided
    scaler = None
    if scaler_path:
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            if verbose:
                print(f"  Scaler: {type(scaler).__name__}")
        else:
            print(f"Warning: Scaler not found at {scaler_path}")

    return model, scaler, model_type, model_name


def load_threshold(threshold_path, model_name=None, verbose=False):
    """
    Load calibrated threshold from JSON file.

    Args:
        threshold_path: Path to threshold_config.json or threshold.txt
        model_name: Model name for finding threshold file
        verbose: Print loading info

    Returns:
        dict: Threshold configuration
    """
    # Try to find threshold config
    threshold_path = Path(threshold_path)

    # If path is a directory, look for threshold files inside
    if threshold_path.is_dir():
        config_file = threshold_path / 'threshold_config.json'
        txt_file = threshold_path / 'threshold.txt'

        if config_file.exists():
            threshold_path = config_file
        elif txt_file.exists():
            threshold_path = txt_file
        else:
            raise FileNotFoundError(
                f"No threshold file found in {threshold_path}\n"
                f"Expected: threshold_config.json or threshold.txt"
            )

    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    if verbose:
        print(f"\nLoading threshold: {threshold_path}")

    # Load threshold configuration
    if threshold_path.suffix == '.json':
        with open(threshold_path, 'r') as f:
            threshold_config = json.load(f)
    elif threshold_path.suffix == '.txt':
        with open(threshold_path, 'r') as f:
            threshold_value = float(f.read().strip())
        threshold_config = {
            'threshold': threshold_value,
            'source': str(threshold_path)
        }
    else:
        raise ValueError(f"Unsupported threshold file format: {threshold_path.suffix}")

    if verbose:
        print(f"  Threshold: {threshold_config['threshold']:.6f}")
        if 'target_far_per_week' in threshold_config:
            print(f"  Target FAR: {threshold_config['target_far_per_week']:.3f}/week")

    return threshold_config


def load_input_data(input_path, verbose=False):
    """
    Load input feature data from CSV or Parquet.

    Args:
        input_path: Path to input file
        verbose: Print loading info

    Returns:
        pd.DataFrame: Input data
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"ERROR: Input file not found: {input_path}\n\n"
            f"Solution: Check that the input path is correct."
        )

    if verbose:
        print(f"\nLoading input data: {input_path}")

    # Load based on file extension
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix} (use .csv or .parquet)")

    if verbose:
        print(f"  Samples: {len(df)}")
        print(f"  Columns: {len(df.columns)}")

    return df


def identify_feature_columns(df, expected_features=None, verbose=False):
    """
    Identify feature columns vs. metadata columns.

    Args:
        df: Input dataframe
        expected_features: List of expected feature names (if known)
        verbose: Print column info

    Returns:
        tuple: (feature_columns, metadata_columns)
    """
    # Common metadata column names
    metadata_cols = [
        'timestamp', 'file_index', 'source_file', 'window_id',
        'sensor', 'split', 'unit', 'cycle', 'fault_type', 'file_name',
        'bearing', 'load', 'fault_size', 'Unnamed: 0', 'index'
    ]

    if expected_features:
        # Use expected features
        feature_cols = [col for col in expected_features if col in df.columns]
        metadata_cols_present = [col for col in df.columns if col not in expected_features]
    else:
        # Infer: numeric columns that aren't metadata
        feature_cols = [
            col for col in df.columns
            if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        metadata_cols_present = [col for col in df.columns if col not in feature_cols]

    if verbose:
        print(f"\nColumn identification:")
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Metadata columns: {len(metadata_cols_present)}")

    return feature_cols, metadata_cols_present


def validate_features(df, expected_features, verbose=False):
    """
    Validate that input data has all required features.

    Args:
        df: Input dataframe
        expected_features: List of expected feature names
        verbose: Print validation details

    Returns:
        tuple: (is_valid, missing_features, extra_features)
    """
    input_features = set(df.columns)
    expected_features_set = set(expected_features)

    missing_features = expected_features_set - input_features
    extra_features = input_features - expected_features_set

    # Remove metadata columns from extra features
    metadata_cols = [
        'timestamp', 'file_index', 'source_file', 'window_id',
        'sensor', 'split', 'unit', 'cycle', 'fault_type', 'file_name',
        'bearing', 'load', 'fault_size', 'Unnamed: 0', 'index'
    ]
    extra_features = extra_features - set(metadata_cols)

    is_valid = len(missing_features) == 0

    if verbose or not is_valid:
        print(f"\nFeature validation:")
        print(f"  Expected: {len(expected_features)} features")
        print(f"  Found: {len(input_features)} columns")

        if missing_features:
            print(f"\n  ❌ Missing features ({len(missing_features)}):")
            for feat in sorted(list(missing_features)[:10]):
                print(f"     - {feat}")
            if len(missing_features) > 10:
                print(f"     ... and {len(missing_features) - 10} more")

        if extra_features and verbose:
            print(f"\n  ⚠️  Extra columns ({len(extra_features)}) - will be preserved as metadata:")
            for feat in sorted(list(extra_features)[:5]):
                print(f"     - {feat}")
            if len(extra_features) > 5:
                print(f"     ... and {len(extra_features) - 5} more")

        if is_valid:
            print(f"  ✓ All required features present")

    if not is_valid:
        raise ValueError(
            f"\nERROR: Input data missing {len(missing_features)} required features.\n\n"
            f"Missing features: {sorted(list(missing_features))[:10]}\n\n"
            f"Solution: Ensure input data has been processed with make_features.py "
            f"using the same configuration as training."
        )

    return is_valid, missing_features, extra_features


def align_features(df, expected_features):
    """
    Align input features to match training order.

    Args:
        df: Input dataframe
        expected_features: List of feature names in training order

    Returns:
        tuple: (X, metadata_df)
    """
    # Extract features in correct order
    X = df[expected_features].values

    # Extract metadata (all other columns)
    metadata_cols = [col for col in df.columns if col not in expected_features]
    if metadata_cols:
        metadata_df = df[metadata_cols].copy()
    else:
        metadata_df = pd.DataFrame(index=df.index)

    # Handle NaN values
    if np.isnan(X).any():
        n_nans = np.isnan(X).sum()
        print(f"\nWarning: Found {n_nans} NaN values, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    return X, metadata_df


def compute_anomaly_scores(model, X, model_type, verbose=False):
    """
    Compute anomaly scores using trained model.

    Args:
        model: Trained anomaly detection model
        X: Feature matrix
        model_type: Model type string
        verbose: Print scoring details

    Returns:
        np.ndarray: Anomaly scores
    """
    if verbose:
        print(f"\nComputing anomaly scores...")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")

    # Compute scores based on model type
    if model_type == 'one_class_svm':
        scores = model.decision_function(X)
    else:
        scores = model.score_samples(X)

    if verbose:
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Score mean: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores


def apply_threshold(scores, threshold, model_type, verbose=False):
    """
    Apply threshold to convert scores to binary alerts.

    Args:
        scores: Anomaly scores
        threshold: Threshold value
        model_type: Model type (affects comparison direction)
        verbose: Print application details

    Returns:
        np.ndarray: Binary alerts (1 = anomaly, 0 = normal)
    """
    # For sklearn models: lower scores = more anomalous
    # Threshold is set such that scores < threshold are anomalies
    is_anomaly = (scores < threshold).astype(int)

    n_anomalies = is_anomaly.sum()
    anomaly_rate = n_anomalies / len(scores)

    if verbose:
        print(f"\nApplying threshold:")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Anomalies detected: {n_anomalies} / {len(scores)} ({anomaly_rate*100:.2f}%)")

    return is_anomaly, n_anomalies, anomaly_rate


def save_predictions(output_path, metadata_df, feature_df, scores, is_anomaly,
                    threshold, model_name, verbose=False):
    """
    Save predictions with metadata to CSV and Parquet.

    Args:
        output_path: Output path (without extension)
        metadata_df: Metadata columns
        feature_df: Original feature columns (optional)
        scores: Anomaly scores
        is_anomaly: Binary alerts
        threshold: Threshold used
        model_name: Name of model used
        verbose: Print save details

    Returns:
        tuple: (csv_path, parquet_path)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine all data
    result_df = metadata_df.copy()

    # Add original features if provided
    if feature_df is not None and len(feature_df.columns) > 0:
        for col in feature_df.columns:
            result_df[col] = feature_df[col].values

    # Add predictions
    result_df['anomaly_score'] = scores
    result_df['is_anomaly'] = is_anomaly
    result_df['threshold_used'] = threshold
    result_df['model_name'] = model_name
    result_df['scored_at'] = datetime.now().isoformat()

    # Save to CSV
    csv_path = output_path.with_suffix('.csv')
    if verbose:
        print(f"\nSaving predictions:")
        print(f"  CSV: {csv_path}")
    result_df.to_csv(csv_path, index=False)

    # Save to Parquet
    parquet_path = output_path.with_suffix('.parquet')
    if verbose:
        print(f"  Parquet: {parquet_path}")
    result_df.to_parquet(parquet_path, index=False)

    return csv_path, parquet_path


def save_metadata(output_path, run_info, model_info, input_info, scoring_results, verbose=False):
    """
    Save run metadata to JSON file.

    Args:
        output_path: Output path (without extension)
        run_info: Run information dict
        model_info: Model information dict
        input_info: Input data information dict
        scoring_results: Scoring results dict
        verbose: Print save details

    Returns:
        Path: Path to metadata JSON file
    """
    output_path = Path(output_path)

    metadata = {
        'run_info': run_info,
        'model_info': model_info,
        'input_info': input_info,
        'scoring_results': scoring_results
    }

    metadata_path = output_path.parent / f"{output_path.name}_metadata.json"

    if verbose:
        print(f"  Metadata: {metadata_path}")

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def get_git_commit():
    """Get current git commit hash for reproducibility."""
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('utf-8').strip()
        return git_hash
    except:
        return 'unknown'


def main():
    """Main entry point for batch scoring."""
    parser = argparse.ArgumentParser(
        description='Score batch data with trained anomaly detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/score_batch.py \\
    --model artifacts/models/ims_iforest.joblib \\
    --input data/features/ims_test.csv \\
    --output artifacts/scores/ims_test_scores

  # With custom threshold override
  python scripts/score_batch.py \\
    --model artifacts/models/cwru_iforest.joblib \\
    --input data/features/cwru_test.csv \\
    --output artifacts/scores/cwru_test_scores \\
    --threshold 0.55

  # With verbose logging
  python scripts/score_batch.py \\
    --model artifacts/models/fd001_iforest.joblib \\
    --input data/features/fd001_test.csv \\
    --output artifacts/scores/fd001_test_scores \\
    --verbose
        """
    )

    # Required arguments
    parser.add_argument('--model', required=True,
                       help='Path to trained model (.joblib or .pth)')
    parser.add_argument('--input', required=True,
                       help='Path to input features (CSV or Parquet)')
    parser.add_argument('--output', required=True,
                       help='Output path for predictions (without extension)')

    # Optional arguments
    parser.add_argument('--scaler',
                       help='Path to scaler (.joblib). If not provided, will look for <model>_scaler.joblib')
    parser.add_argument('--threshold', type=float,
                       help='Override calibrated threshold (optional)')
    parser.add_argument('--threshold-file',
                       help='Path to threshold config file (threshold_config.json or threshold.txt)')
    parser.add_argument('--features-file',
                       help='Path to features.txt file listing expected features')
    parser.add_argument('--batch-size', type=int,
                       help='Process data in chunks (for large files, not yet implemented)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable detailed logging')

    args = parser.parse_args()

    # Start timer
    start_time = time.time()

    # Print header
    print("="*60)
    print("BATCH SCORING - STAGE 6")
    print("="*60)

    # Load model and scaler
    model_path = Path(args.model)

    # Auto-detect scaler path if not provided
    if args.scaler:
        scaler_path = args.scaler
    else:
        # Look for <model_name>_scaler.joblib
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.joblib"
        if not scaler_path.exists():
            scaler_path = None

    model, scaler, model_type, model_name = load_model(
        model_path, scaler_path, verbose=args.verbose
    )

    # Load or infer threshold
    if args.threshold is not None:
        # Use CLI override
        threshold_value = args.threshold
        threshold_config = {
            'threshold': threshold_value,
            'source': 'cli_override'
        }
        if args.verbose:
            print(f"\nUsing threshold from CLI: {threshold_value:.6f}")
    elif args.threshold_file:
        # Load from specified file
        threshold_config = load_threshold(args.threshold_file, verbose=args.verbose)
        threshold_value = threshold_config['threshold']
    else:
        # Try to find threshold in report directory
        report_dir = model_path.parent.parent / 'reports' / model_path.stem
        threshold_file = report_dir / 'threshold_config.json'

        if threshold_file.exists():
            threshold_config = load_threshold(threshold_file, verbose=args.verbose)
            threshold_value = threshold_config['threshold']
        else:
            # No threshold found - use default (0.0 for sklearn models)
            threshold_value = 0.0
            threshold_config = {
                'threshold': threshold_value,
                'source': 'default'
            }
            print(f"\nWarning: No threshold file found, using default: {threshold_value:.6f}")

    # Load expected features if provided
    expected_features = None
    if args.features_file:
        with open(args.features_file, 'r') as f:
            expected_features = [line.strip() for line in f if line.strip()]
    else:
        # Try to find features.txt in report directory
        report_dir = model_path.parent.parent / 'reports' / model_path.stem
        features_file = report_dir / 'features.txt'

        if features_file.exists():
            if args.verbose:
                print(f"\nLoading features from: {features_file}")
            with open(features_file, 'r') as f:
                expected_features = [line.strip() for line in f if line.strip()]

    # Load input data
    df = load_input_data(args.input, verbose=args.verbose)

    # Identify feature vs metadata columns
    if expected_features:
        # Validate features
        validate_features(df, expected_features, verbose=args.verbose)

        # Align features
        X, metadata_df = align_features(df, expected_features)
        feature_names = expected_features
    else:
        # Infer features (all numeric non-metadata columns)
        feature_cols, metadata_cols = identify_feature_columns(df, verbose=args.verbose)
        X = df[feature_cols].values
        metadata_df = df[metadata_cols] if metadata_cols else pd.DataFrame(index=df.index)
        feature_names = feature_cols

        print(f"\nWarning: Feature list not provided, inferring {len(feature_cols)} features")

    if args.verbose:
        print(f"\nFeature matrix shape: {X.shape}")

    # Apply scaling if scaler provided
    if scaler is not None:
        if args.verbose:
            print("Applying feature scaling...")
        X = scaler.transform(X)

    # Compute anomaly scores
    scores = compute_anomaly_scores(model, X, model_type, verbose=args.verbose)

    # Apply threshold
    is_anomaly, n_anomalies, anomaly_rate = apply_threshold(
        scores, threshold_value, model_type, verbose=args.verbose
    )

    # Create feature dataframe (optional - only if we want to include features in output)
    # For now, we'll only include metadata and predictions
    feature_df = None

    # Save predictions
    csv_path, parquet_path = save_predictions(
        args.output, metadata_df, feature_df, scores, is_anomaly,
        threshold_value, model_name, verbose=args.verbose
    )

    # Calculate processing time
    processing_time = time.time() - start_time

    # Prepare metadata
    run_info = {
        'timestamp': datetime.now().isoformat(),
        'script_version': '1.0',
        'git_commit': get_git_commit()
    }

    model_info = {
        'model_path': str(model_path),
        'model_name': model_name,
        'model_type': model_type,
        'scaler_used': scaler is not None,
        'scaler_path': str(scaler_path) if scaler_path else None,
        'n_features': len(feature_names),
        'features_used': feature_names if len(feature_names) <= 100 else feature_names[:100] + ['...']
    }

    input_info = {
        'input_path': str(args.input),
        'total_samples': int(len(df)),
        'file_size_mb': round(Path(args.input).stat().st_size / (1024 * 1024), 2)
    }

    # Add date range if timestamp column exists
    if 'timestamp' in metadata_df.columns:
        try:
            timestamps = pd.to_datetime(metadata_df['timestamp'])
            input_info['date_range'] = f"{timestamps.min()} to {timestamps.max()}"
        except:
            pass

    scoring_results = {
        'threshold_used': float(threshold_value),
        'threshold_source': threshold_config.get('source', 'unknown'),
        'anomalies_detected': int(n_anomalies),
        'anomaly_rate': float(anomaly_rate),
        'processing_time_seconds': round(processing_time, 2)
    }

    # Save metadata
    metadata_path = save_metadata(
        args.output, run_info, model_info, input_info, scoring_results,
        verbose=args.verbose
    )

    # Print summary
    print("\n" + "="*60)
    print("SCORING COMPLETE")
    print("="*60)
    print(f"\nProcessed: {len(df):,} samples")
    print(f"Anomalies detected: {n_anomalies:,} ({anomaly_rate*100:.2f}%)")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"\nOutputs:")
    print(f"  Predictions: {csv_path}")
    print(f"  Predictions: {parquet_path}")
    print(f"  Metadata: {metadata_path}")
    print("="*60)


if __name__ == '__main__':
    main()
