#!/usr/bin/env python3
"""
Model Training Script - Stage 3

Trains anomaly detection models on extracted features.
Supports: Isolation Forest, kNN-LOF, One-Class SVM

Usage: python scripts/train.py --config configs/models/isolation_forest.yaml
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings('ignore', category=FutureWarning)


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_scaler(scaler_type):
    """Get scaler instance based on type"""
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    elif scaler_type == 'robust':
        return RobustScaler()
    elif scaler_type == 'none' or scaler_type is None:
        return None
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")


def load_feature_data(dataset_config):
    """Load feature data from parquet file"""
    features_path = Path(dataset_config['paths']['features_output_path'])
    print(f"Loading features: {features_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_parquet(features_path)
    print(f"Loaded {len(df)} feature vectors with {len(df.columns)} columns")

    return df


def filter_normal_baseline(df, dataset_config):
    """
    Filter data to only include normal baseline samples for training.
    For IMS: use only first N files (healthy bearing operation)
    """
    paths = dataset_config.get('paths', {})

    # Check if this is IMS dataset with normal baseline setting
    if 'raw_input_dir' in paths:  # IMS dataset
        anomaly_config = dataset_config.get('prep', {}).get('anomaly_detection', {})

        if anomaly_config.get('use_early_as_normal', False):
            normal_baseline_files = anomaly_config.get('normal_baseline_files')

            if normal_baseline_files is not None:
                print(f"Filtering to first {normal_baseline_files} files (normal baseline)")

                # Filter by file_index
                if 'file_index' in df.columns:
                    df_normal = df[df['file_index'] < normal_baseline_files].copy()
                    print(f"Normal baseline: {len(df_normal)} samples from {df_normal['file_index'].nunique()} files")
                    return df_normal

    # For other datasets or if no baseline filtering, return all data
    print("No normal baseline filtering - using all data")
    return df


def select_feature_columns(df, model_config, dataset_config):
    """
    Select feature columns for training.
    Exclude metadata columns (timestamp, file_index, source_file, etc.)
    """
    # Columns to exclude (metadata)
    metadata_cols = ['timestamp', 'file_index', 'source_file', 'window_id', 'sensor',
                     'split', 'unit', 'cycle', 'fault_type', 'fault_size_mils',
                     'load_hp', 'sensor_location']

    # Get feature specification from model config
    features_spec = model_config.get('features', 'all')

    if features_spec == 'all':
        # Use all numeric columns except metadata
        feature_cols = [col for col in df.columns if col not in metadata_cols]

        # Additional filtering: only numeric columns
        feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    else:
        # Use specified feature list
        feature_cols = features_spec

    print(f"Selected {len(feature_cols)} feature columns")
    print(f"Features: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Features: {feature_cols}")

    return feature_cols


def train_isolation_forest(X_train, hyperparams):
    """Train Isolation Forest model"""
    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=hyperparams.get('n_estimators', 100),
        contamination=hyperparams.get('contamination', 0.1),
        max_features=hyperparams.get('max_features', 1.0),
        bootstrap=hyperparams.get('bootstrap', False),
        max_samples=hyperparams.get('max_samples', 'auto'),
        random_state=hyperparams.get('random_state', 42),
        n_jobs=-1,  # Use all cores
        verbose=1
    )

    model.fit(X_train)
    print("Training complete!")

    return model


def train_knn_lof(X_train, hyperparams):
    """Train kNN Local Outlier Factor model"""
    print("\nTraining kNN-LOF...")

    model = LocalOutlierFactor(
        n_neighbors=hyperparams.get('n_neighbors', 20),
        contamination=hyperparams.get('contamination', 0.1),
        algorithm=hyperparams.get('algorithm', 'auto'),
        leaf_size=hyperparams.get('leaf_size', 30),
        metric=hyperparams.get('metric', 'minkowski'),
        p=hyperparams.get('p', 2),
        novelty=True,  # Required for prediction on new data
        n_jobs=-1
    )

    model.fit(X_train)
    print("Training complete!")

    return model


def train_one_class_svm(X_train, hyperparams):
    """Train One-Class SVM model"""
    print("\nTraining One-Class SVM...")

    model = OneClassSVM(
        kernel=hyperparams.get('kernel', 'rbf'),
        nu=hyperparams.get('nu', 0.1),
        gamma=hyperparams.get('gamma', 'scale'),
        degree=hyperparams.get('degree', 3),
        coef0=hyperparams.get('coef0', 0.0),
        tol=hyperparams.get('tol', 0.001),
        shrinking=hyperparams.get('shrinking', True),
        cache_size=hyperparams.get('cache_size', 200),
        max_iter=hyperparams.get('max_iter', -1)
    )

    model.fit(X_train)
    print("Training complete!")

    return model


def evaluate_on_training_data(model, X_train, model_type):
    """Evaluate model on training data to check sanity"""
    print("\nEvaluating on training data...")

    if model_type == 'isolation_forest':
        # Predict returns -1 for anomalies, 1 for normal
        predictions = model.predict(X_train)
        scores = model.score_samples(X_train)
    elif model_type == 'knn_lof':
        predictions = model.predict(X_train)
        scores = model.score_samples(X_train)
    elif model_type == 'one_class_svm':
        predictions = model.predict(X_train)
        scores = model.decision_function(X_train)
    else:
        return None

    # Count normal (1) vs anomaly (-1)
    n_normal = np.sum(predictions == 1)
    n_anomaly = np.sum(predictions == -1)
    anomaly_rate = n_anomaly / len(predictions)

    print(f"Training data predictions:")
    print(f"  Normal: {n_normal} ({100*n_normal/len(predictions):.1f}%)")
    print(f"  Anomaly: {n_anomaly} ({100*anomaly_rate:.1f}%)")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Score mean: {scores.mean():.4f}, std: {scores.std():.4f}")

    return {
        'n_normal': int(n_normal),
        'n_anomaly': int(n_anomaly),
        'anomaly_rate': float(anomaly_rate),
        'score_min': float(scores.min()),
        'score_max': float(scores.max()),
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std())
    }


def save_model_and_metadata(model, scaler, model_config, dataset_config,
                            feature_cols, training_stats, config_path):
    """Save trained model, scaler, and run metadata"""
    # Create output directories
    model_path = Path(model_config['paths']['model_output'])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    report_dir = Path(model_config['paths']['report_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    print(f"\nSaving model: {model_path}")
    joblib.dump(model, model_path)

    # Save scaler if present
    if scaler is not None:
        scaler_path = Path(model_config['paths']['scaler_output'])
        print(f"Saving scaler: {scaler_path}")
        joblib.dump(scaler, scaler_path)

    # Get git commit hash
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except:
        git_hash = 'unknown'

    # Create run metadata
    run_metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_config['model_name'],
        'model_type': model_config['model_type'],
        'dataset_config': str(config_path),
        'git_commit': git_hash,
        'hyperparameters': model_config['hyperparameters'],
        'scaler': model_config.get('scaler'),
        'n_features': len(feature_cols),
        'features': feature_cols,
        'n_training_samples': training_stats.get('n_samples'),
        'training_evaluation': training_stats.get('evaluation'),
        'random_state': model_config.get('random_state', 42)
    }

    # Save run metadata
    run_json_path = report_dir / 'run.json'
    print(f"Saving run metadata: {run_json_path}")
    with open(run_json_path, 'w') as f:
        json.dump(run_metadata, f, indent=2)

    # Save feature list
    features_path = report_dir / 'features.txt'
    print(f"Saving feature list: {features_path}")
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    print("\n" + "="*60)
    print("Model training complete!")
    print(f"Model saved: {model_path}")
    print(f"Report dir: {report_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    parser.add_argument('--config', required=True, help='Model config YAML file')
    args = parser.parse_args()

    print("="*60)
    print("ANOMALY DETECTION MODEL TRAINING - STAGE 3")
    print("="*60)

    # Load model config
    print(f"\nLoading model config: {args.config}")
    model_config = load_config(args.config)

    # Load dataset config
    dataset_config_path = model_config['dataset_config']
    print(f"Loading dataset config: {dataset_config_path}")
    dataset_config = load_config(dataset_config_path)

    # Load feature data
    df = load_feature_data(dataset_config)

    # Filter to normal baseline (for IMS)
    df_train = filter_normal_baseline(df, dataset_config)

    # Select feature columns
    feature_cols = select_feature_columns(df_train, model_config, dataset_config)
    X_train = df_train[feature_cols].values

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")

    # Handle missing values
    if np.isnan(X_train).any():
        print("Warning: Found NaN values, filling with 0")
        X_train = np.nan_to_num(X_train, nan=0.0)

    # Scale features
    scaler_type = model_config.get('scaler')
    scaler = get_scaler(scaler_type)

    if scaler is not None:
        print(f"\nApplying {scaler_type} scaling...")
        X_train = scaler.fit_transform(X_train)

    # Train model based on type
    model_type = model_config['model_type']
    hyperparams = model_config.get('hyperparameters', {})

    if model_type == 'isolation_forest':
        model = train_isolation_forest(X_train, hyperparams)
    elif model_type == 'knn_lof':
        model = train_knn_lof(X_train, hyperparams)
    elif model_type == 'one_class_svm':
        model = train_one_class_svm(X_train, hyperparams)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Evaluate on training data
    eval_stats = evaluate_on_training_data(model, X_train, model_type)

    # Save model and metadata
    training_stats = {
        'n_samples': X_train.shape[0],
        'n_features': X_train.shape[1],
        'evaluation': eval_stats
    }

    save_model_and_metadata(
        model, scaler, model_config, dataset_config,
        feature_cols, training_stats, args.config
    )


if __name__ == '__main__':
    main()
