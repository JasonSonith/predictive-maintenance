#!/usr/bin/env python3
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from numpy._typing import _128Bit
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import OneClassSVM
warnings.filterwarnings('ignore', category=FutureWarning)

# PyTorch imports (optional - only needed for AutoEncoder)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None


class AutoEncoder:
    """
    PyTorch-based Autoencoder for anomaly detection.

    Uses reconstruction error to detect anomalies:
    - Normal data: Low reconstruction error (model learned the pattern)
    - Anomalous data: High reconstruction error (model can't reconstruct it well)

    Implements sklearn-like API: fit(), predict(), score_samples()
    """

    def __init__(self, encoder_dims=[64, 32, 16], bottleneck_dim=8,
                 epochs=50, batch_size=32, learning_rate=0.001,
                 dropout=0.2, weight_decay=0.0001,
                 early_stopping_patience=10, validation_split=0.1,
                 contamination=0.1, activation='relu', loss='mse',
                 optimizer='adam', device='cpu', random_state=42):
        """
        Args:
            encoder_dims: List of hidden layer sizes for encoder
            bottleneck_dim: Size of compressed representation
            epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            dropout: Dropout rate for regularization
            weight_decay: L2 regularization strength
            early_stopping_patience: Stop if no improvement for N epochs
            validation_split: Fraction of data for validation
            contamination: Expected proportion of anomalies (for threshold)
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            loss: Loss function ('mse', 'mae')
            optimizer: Optimizer ('adam', 'sgd', 'rmsprop')
            device: 'cpu' or 'cuda'
            random_state: Random seed
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AutoEncoder. Install with: pip install torch")

        self.encoder_dims = encoder_dims
        self.bottleneck_dim = bottleneck_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.contamination = contamination
        self.activation = activation
        self.loss_fn_name = loss
        self.optimizer_name = optimizer
        self.device = device
        self.random_state = random_state

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # These will be set during fit()
        self.model = None
        self.threshold = None
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.input_dim = None

    def _build_model(self, input_dim):
        """Build the encoder-decoder network"""
        layers = []

        # Encoder
        prev_dim = input_dim
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

        # Output layer (no activation - we want raw reconstruction)
        layers.append(nn.Linear(prev_dim, input_dim))

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

    def _get_loss_fn(self):
        """Get loss function"""
        if self.loss_fn_name == 'mse':
            return nn.MSELoss()
        elif self.loss_fn_name == 'mae':
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss: {self.loss_fn_name}")

    def _get_optimizer(self):
        """Get optimizer"""
        if self.optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(),
                            lr=self.learning_rate,
                            weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(),
                           lr=self.learning_rate,
                           weight_decay=self.weight_decay)
        elif self.optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def fit(self, X):
        """
        Train the autoencoder on normal data.

        Args:
            X: Training data (numpy array, shape [n_samples, n_features])
        """
        X = np.array(X, dtype=np.float32)
        n_samples, self.input_dim = X.shape

        # Split into train and validation
        n_val = int(n_samples * self.validation_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train = X[train_indices]
        X_val = X[val_indices]

        print(f"\nTraining samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Using device: {self.device}")

        # Build model
        self.model = self._build_model(self.input_dim)
        self.model.to(self.device)

        print(f"\nTraining autoencoder...")
        print("Model architecture:")
        print(self.model)

        # Setup training
        criterion = self._get_loss_fn()
        optimizer = self._get_optimizer()

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(X_train)  # Target is same as input
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(X_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining for up to {self.epochs} epochs (early stopping patience: {self.early_stopping_patience})...")

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_X)

            train_loss /= len(X_train)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * len(batch_X)

            val_loss /= len(X_val)

            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {self.early_stopping_patience} epochs)")
                    break

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")

        # Set anomaly threshold based on training data reconstruction errors
        reconstruction_errors = self._compute_reconstruction_errors(X)
        threshold_percentile = 100 * (1 - self.contamination)
        self.threshold = np.percentile(reconstruction_errors, threshold_percentile)

        print(f"\nAnomaly threshold set at: {self.threshold:.6f} (contamination: {self.contamination})")

        return self

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

        Lower (more negative) scores = more anomalous
        This matches sklearn convention.

        Args:
            X: Data to score (numpy array)

        Returns:
            scores: Anomaly scores (numpy array)
        """
        errors = self._compute_reconstruction_errors(X)
        # Return negative errors to match sklearn convention (lower = more anomalous)
        return -errors

    def predict(self, X):
        """
        Predict anomaly labels.

        Returns:
            labels: 1 for normal, -1 for anomaly
        """
        errors = self._compute_reconstruction_errors(X)
        predictions = np.where(errors > self.threshold, -1, 1)
        return predictions


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
        raise NotImplementedError("Subclass must implement _create_model()")

    def train(self):
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

        # Print threshold for AutoEncoder
        if isinstance(self.model, AutoEncoder):
            print(f"  Reconstruction error threshold: {self.model.threshold:.6f}")

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

        # Save model (PyTorch models use .pth, sklearn models use .joblib)
        print(f"\nSaving model: {model_path}")
        if isinstance(self.model, AutoEncoder):
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.model.state_dict(),
                'threshold': self.model.threshold,
                'input_dim': self.model.input_dim,
                'encoder_dims': self.model.encoder_dims,
                'bottleneck_dim': self.model.bottleneck_dim,
                'activation': self.model.activation,
                'dropout': self.model.dropout,
                'training_history': self.model.training_history
            }, model_path)
        else:
            # Save sklearn model
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

        # Add training history for AutoEncoder
        if isinstance(self.model, AutoEncoder):
            run_metadata['training_history'] = self.model.training_history

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

class IsolationForestTrainer(ModelTrainer):
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

class AutoEncoderTrainer(ModelTrainer):
    def _create_model(self):
        return AutoEncoder(
            encoder_dims=self.hyperparams.get('encoder_dims', [64, 32, 16]),
            bottleneck_dim=self.hyperparams.get('bottleneck_dim', 8),
            epochs=self.hyperparams.get('epochs', 50),
            batch_size=self.hyperparams.get('batch_size', 32),
            learning_rate=self.hyperparams.get('learning_rate', 0.001),
            dropout=self.hyperparams.get('dropout', 0.2),
            weight_decay=self.hyperparams.get('weight_decay', 0.0001),
            early_stopping_patience=self.hyperparams.get('early_stopping_patience', 10),
            validation_split=self.hyperparams.get('validation_split', 0.1),
            contamination=self.hyperparams.get('contamination', 0.1),
            activation=self.hyperparams.get('activation', 'relu'),
            loss=self.hyperparams.get('loss', 'mse'),
            optimizer=self.hyperparams.get('optimizer', 'adam'),
            device=self.model_config.get('device', 'cpu'),
            random_state=self.model_config.get('random_state', 42)
        )

MODEL_TRAINERS = {
    'isolation_forest': IsolationForestTrainer,
    'knn_lof': KNNLOFTrainer,
    'one_class_svm': OneClassSVMTrainer,
    'autoencoder': AutoEncoderTrainer
}

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
