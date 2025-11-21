#!/usr/bin/env python3
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

#Base class for loading data, preparing, evaluating, and saving
class ModelTrainer:

    def __init__(self, config_path):
        """
        Constructor: Sets up everything we need before training.
        This runs automatically when you create a trainer.

        Args:
            config_path: Path to the model config YAML file
        """
        print("="*60)
        print("ANOMALY DETECTION MODEL TRAINING - STAGE 3")
        print("="*60)

        # Load configurations
        print(f"\nLoading model config: {config_path}")
        self.config_path = config_path
        self.model_config = self._load_yaml(config_path)

        dataset_config_path = self.model_config['dataset_config']
        print(f"Loading dataset config: {dataset_config_path}")
        self.dataset_config = self._load_yaml(dataset_config_path)

        # Store important info
        self.model_type = self.model_config['model_type']
        self.hyperparams = self.model_config.get('hyperparameters', {})

        # These will be set during training
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.X_train = None

    def _load_yaml(self, path):
        """Helper: Load a YAML config file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self):
        """
        Step 1: Load the feature data from disk.
        Returns a pandas DataFrame with all the features.
        """
        features_path = Path(self.dataset_config['paths']['features_output_path'])
        print(f"\nLoading features: {features_path}")

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        df = pd.read_parquet(features_path)
        print(f"Loaded {len(df)} feature vectors with {len(df.columns)} columns")
        return df

    def filter_normal_baseline(self, df):
        """
        Step 2: Filter to only "normal" data for training (IMS datasets).

        For IMS, we only train on the first N files (healthy bearings).
        Later files show degradation and are used for testing only.
        """
        paths = self.dataset_config.get('paths', {})

        # Check if this is IMS dataset with normal baseline setting
        if 'raw_input_dir' in paths:  # IMS dataset
            anomaly_config = self.dataset_config.get('prep', {}).get('anomaly_detection', {})

            if anomaly_config.get('use_early_as_normal', False):
                normal_baseline_files = anomaly_config.get('normal_baseline_files')

                if normal_baseline_files is not None:
                    print(f"Filtering to first {normal_baseline_files} files (normal baseline)")

                    if 'file_index' in df.columns:
                        df_normal = df[df['file_index'] < normal_baseline_files].copy()
                        print(f"Normal baseline: {len(df_normal)} samples from {df_normal['file_index'].nunique()} files")
                        return df_normal

        # For other datasets or if no baseline filtering, return all data
        print("No normal baseline filtering - using all data")
        return df

    def select_features(self, df):
        """
        Step 3: Pick which columns are features (vs metadata like timestamps).

        We don't want to train on things like 'timestamp' or 'file_index'.
        Only use numeric features like 'mean', 'std', 'rms', etc.
        """
        # These are metadata, not features
        metadata_cols = ['timestamp', 'file_index', 'source_file', 'window_id', 'sensor',
                         'split', 'unit', 'cycle', 'fault_type', 'fault_size_mils',
                         'load_hp', 'sensor_location']

        # Get feature specification from config
        features_spec = self.model_config.get('features', 'all')

        if features_spec == 'all':
            # Use all numeric columns except metadata
            feature_cols = [col for col in df.columns if col not in metadata_cols]
            feature_cols = [col for col in feature_cols
                           if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        else:
            # Use specified feature list from config
            feature_cols = features_spec

        print(f"\nSelected {len(feature_cols)} feature columns")
        if len(feature_cols) > 10:
            print(f"Features: {feature_cols[:10]}...")
        else:
            print(f"Features: {feature_cols}")

        return feature_cols

    def prepare_data(self):
        """
        Step 4: Load and prepare all the data for training.
        This combines all the preparation steps.
        """
        # Load features
        df = self.load_data()

        # Filter to normal baseline (for IMS)
        df_train = self.filter_normal_baseline(df)

        # Select feature columns
        self.feature_cols = self.select_features(df_train)
        X_train = df_train[self.feature_cols].values

        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")

        # Handle missing values
        if np.isnan(X_train).any():
            print("Warning: Found NaN values, filling with 0")
            X_train = np.nan_to_num(X_train, nan=0.0)

        # Scale features (normalize the data)
        scaler_type = self.model_config.get('scaler')
        self.scaler = self._get_scaler(scaler_type)

        if self.scaler is not None:
            print(f"\nApplying {scaler_type} scaling...")
            X_train = self.scaler.fit_transform(X_train)

        self.X_train = X_train
        return X_train

    def _get_scaler(self, scaler_type):
        """Helper: Create a scaler based on the config"""
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

    def _create_model(self):
        """
        THIS IS THE KEY METHOD!

        This is left blank here - each specific model class fills this in.
        For example, IsolationForestTrainer says "create an IsolationForest".
        """
        raise NotImplementedError("Subclass must implement _create_model()")

    def train(self):
        """
        Step 5: The main training process.
        This orchestrates everything: prepare data, create model, train, evaluate, save.
        """
        # Prepare data
        X_train = self.prepare_data()

        # Create the model (each subclass defines this)
        print(f"\nTraining {self.model_type}...")
        self.model = self._create_model()

        # Train the model
        self.model.fit(X_train)
        print("Training complete!")

        # Evaluate
        eval_stats = self.evaluate()

        # Save everything
        training_stats = {
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'evaluation': eval_stats
        }
        self.save(training_stats)

    def evaluate(self):
        """
        Step 6: Evaluate the model on training data.
        This checks if the model is working correctly.
        """
        print("\nEvaluating on training data...")

        # Get predictions and scores
        predictions = self.model.predict(self.X_train)

        # Get anomaly scores (method depends on model type)
        if self.model_type == 'one_class_svm':
            scores = self.model.decision_function(self.X_train)
        else:
            scores = self.model.score_samples(self.X_train)

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

    def save(self, training_stats):
        """
        Step 7: Save the trained model and all metadata.
        """
        # Create output directories
        model_path = Path(self.model_config['paths']['model_output'])
        model_path.parent.mkdir(parents=True, exist_ok=True)

        report_dir = Path(self.model_config['paths']['report_dir'])
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        print(f"\nSaving model: {model_path}")
        joblib.dump(self.model, model_path)

        # Save scaler if present
        if self.scaler is not None:
            scaler_path = Path(self.model_config['paths']['scaler_output'])
            print(f"Saving scaler: {scaler_path}")
            joblib.dump(self.scaler, scaler_path)

        # Get git commit hash for reproducibility
        try:
            import subprocess
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        except:
            git_hash = 'unknown'

        # Create run metadata
        run_metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_config['model_name'],
            'model_type': self.model_config['model_type'],
            'dataset_config': str(self.config_path),
            'git_commit': git_hash,
            'hyperparameters': self.model_config['hyperparameters'],
            'scaler': self.model_config.get('scaler'),
            'n_features': len(self.feature_cols),
            'features': self.feature_cols,
            'n_training_samples': training_stats.get('n_samples'),
            'training_evaluation': training_stats.get('evaluation'),
            'random_state': self.model_config.get('random_state', 42)
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
            f.write('\n'.join(self.feature_cols))

        print("\n" + "="*60)
        print("Model training complete!")
        print(f"Model saved: {model_path}")
        print(f"Report dir: {report_dir}")
        print("="*60)


# ============================================================================
# MODEL-SPECIFIC CLASSES
# These are the "toppings" for each cookie type
# Each one just says "here's my model" - everything else is inherited
# ============================================================================

class IsolationForestTrainer(ModelTrainer):
    """
    Trainer for Isolation Forest model.
    Only needs to define how to create the model - that's it!
    """
    def _create_model(self):
        return IsolationForest(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            contamination=self.hyperparams.get('contamination', 0.1),
            max_features=self.hyperparams.get('max_features', 1.0),
            bootstrap=self.hyperparams.get('bootstrap', False),
            max_samples=self.hyperparams.get('max_samples', 'auto'),
            random_state=self.hyperparams.get('random_state', 42),
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )


class KNNLOFTrainer(ModelTrainer):
    """
    Trainer for k-Nearest Neighbors Local Outlier Factor model.
    Again, just defines the model creation - everything else is inherited!
    """
    def _create_model(self):
        return LocalOutlierFactor(
            n_neighbors=self.hyperparams.get('n_neighbors', 20),
            contamination=self.hyperparams.get('contamination', 0.1),
            algorithm=self.hyperparams.get('algorithm', 'auto'),
            leaf_size=self.hyperparams.get('leaf_size', 30),
            metric=self.hyperparams.get('metric', 'minkowski'),
            p=self.hyperparams.get('p', 2),
            novelty=True,  # Required for prediction on new data
            n_jobs=-1
        )


class OneClassSVMTrainer(ModelTrainer):
    """
    Trainer for One-Class SVM model.
    Same pattern - just the model creation!
    """
    def _create_model(self):
        return OneClassSVM(
            kernel=self.hyperparams.get('kernel', 'rbf'),
            nu=self.hyperparams.get('nu', 0.1),
            gamma=self.hyperparams.get('gamma', 'scale'),
            degree=self.hyperparams.get('degree', 3),
            coef0=self.hyperparams.get('coef0', 0.0),
            tol=self.hyperparams.get('tol', 0.001),
            shrinking=self.hyperparams.get('shrinking', True),
            cache_size=self.hyperparams.get('cache_size', 200),
            max_iter=self.hyperparams.get('max_iter', -1)
        )


# ============================================================================
# FACTORY PATTERN
# This is the "menu" that maps model names to trainer classes
# ============================================================================

# This dictionary is the magic selector
# It says: "if model type is X, use trainer Y"
MODEL_TRAINERS = {
    'isolation_forest': IsolationForestTrainer,
    'knn_lof': KNNLOFTrainer,
    'one_class_svm': OneClassSVMTrainer
}


# ============================================================================
# MAIN FUNCTION
# Look how simple this is now!
# ============================================================================

def main():
    """
    Main entry point - now super simple!
    1. Parse command line arguments
    2. Load config to see which model type
    3. Create the right trainer
    4. Train!
    """
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    parser.add_argument('--config', required=True, help='Model config YAML file')
    args = parser.parse_args()

    # Load config to get model type
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model_type']

    # Check if we support this model
    if model_type not in MODEL_TRAINERS:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: {list(MODEL_TRAINERS.keys())}")

    # Create the trainer (this picks the right class from the dictionary)
    trainer_class = MODEL_TRAINERS[model_type]
    trainer = trainer_class(args.config)

    # Train! (This calls all the methods: prepare, create, train, evaluate, save)
    trainer.train()


if __name__ == '__main__':
    main()
