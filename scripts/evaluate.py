#!/usr/bin/env python3
"""
Model Evaluation - Stage 5

Evaluates trained anomaly detection models on test data and generates comprehensive
reports including metrics, visualizations, and SHAP explanations.

This script:
1. Loads trained model, threshold config, and dataset config
2. Generates predictions on test data (or train/val if specified)
3. Computes dataset-adaptive metrics (unsupervised vs supervised)
4. Creates 7 types of visualizations
5. Generates SHAP explanations (global + individual)
6. Assembles comprehensive HTML report
7. Saves all artifacts for reproducibility

Usage:
    # Simple usage - auto-detect all files from report directory
    python scripts/evaluate.py --report_dir artifacts/reports/ims_iforest/

    # Advanced usage - specify components individually
    python scripts/evaluate.py \
        --model artifacts/models/ims_iforest/model.joblib \
        --threshold artifacts/reports/ims_iforest/threshold_config.json \
        --config configs/ims.yaml \
        --output artifacts/reports/ims_iforest/
"""

import argparse
import base64
import json
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# PyTorch imports (optional - for AutoEncoder)
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None

# SHAP imports (optional but recommended)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    print("Warning: SHAP not available. Install with: pip install shap")


class AutoEncoder:
    """PyTorch AutoEncoder wrapper for loading saved models."""
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

        self.model = self._build_model()
        self.model.to(device)
        self.model.eval()

    def _build_model(self):
        """Build encoder-decoder network"""
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

        # Decoder
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
        """Compute anomaly scores (negative reconstruction errors)"""
        errors = self._compute_reconstruction_errors(X)
        return -errors


class ModelEvaluator:
    """Main evaluation orchestrator"""

    def __init__(self, model_path, threshold_config, dataset_config, output_dir,
                 feature_names=None):
        """
        Args:
            model_path: Path to trained model (.joblib or .pth)
            threshold_config: Dictionary with threshold configuration
            dataset_config: Dictionary with dataset configuration
            output_dir: Directory for saving evaluation outputs
            feature_names: List of feature names (if None, loaded from config)
        """
        self.model_path = Path(model_path)
        self.threshold_config = threshold_config
        self.dataset_config = dataset_config
        self.output_dir = Path(output_dir)
        self.feature_names = feature_names

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and scaler
        self.model, self.scaler = self._load_model()

        # Store dataset info
        self.dataset_name = dataset_config.get('dataset_name', 'unknown')
        self.features_path = Path(dataset_config['paths']['features_output_path'])

    def _load_model(self):
        """Load trained model and scaler"""
        print(f"\nLoading model: {self.model_path}")

        # Detect model type
        if str(self.model_path).endswith('.pth'):
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch required for AutoEncoder models")

            # Load AutoEncoder
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            model = AutoEncoder(
                input_dim=checkpoint['input_dim'],
                encoder_dims=checkpoint['encoder_dims'],
                bottleneck_dim=checkpoint['bottleneck_dim'],
                activation=checkpoint['activation'],
                dropout=checkpoint['dropout'],
                device='cpu'
            )
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.threshold = checkpoint['threshold']
            print(f"Loaded AutoEncoder (input_dim={checkpoint['input_dim']})")
        else:
            # Load sklearn model
            model = joblib.load(self.model_path)
            print(f"Loaded {type(model).__name__}")

        # Load scaler if exists
        scaler = None
        scaler_path = self.model_path.parent / f"{self.model_path.stem}_scaler.joblib"
        if scaler_path.exists():
            print(f"Loading scaler: {scaler_path}")
            scaler = joblib.load(scaler_path)

        return model, scaler

    def load_data(self, split='test'):
        """Load feature data for specified split"""
        print(f"\nLoading features: {self.features_path}")
        df = pd.read_parquet(self.features_path)

        # Filter to split
        if 'split' in df.columns:
            df_split = df[df['split'] == split].copy()
            print(f"Filtered to split='{split}': {len(df_split)} samples")

            if len(df_split) == 0:
                raise ValueError(f"No samples found for split='{split}'")

            return df_split
        else:
            print(f"No 'split' column found, using all data: {len(df)} samples")
            return df

    def prepare_features(self, df):
        """Extract and scale features from dataframe"""
        # Select feature columns
        if self.feature_names is None:
            # Use all numeric columns except metadata
            metadata_cols = ['timestamp', 'file_index', 'source_file', 'window_id',
                            'sensor', 'split', 'unit', 'cycle', 'fault_type',
                            'fault_size_mils', 'load_hp', 'sensor_location']
            self.feature_names = [col for col in df.columns if col not in metadata_cols]
            self.feature_names = [col for col in self.feature_names
                                 if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

        print(f"Using {len(self.feature_names)} features")

        # Extract feature matrix
        X = df[self.feature_names].values

        # Handle NaN
        if np.isnan(X).any():
            print("Warning: Found NaN values, filling with 0")
            X = np.nan_to_num(X, nan=0.0)

        # Apply scaling
        if self.scaler is not None:
            print("Applying feature scaling...")
            X = self.scaler.transform(X)

        return X

    def predict(self, X):
        """Generate anomaly scores and binary predictions"""
        print("\nGenerating predictions...")

        # Get anomaly scores
        model_class = type(self.model).__name__
        if 'SVM' in model_class:
            scores = self.model.decision_function(X)
        else:
            scores = self.model.score_samples(X)

        # Apply threshold to get binary predictions
        threshold = self.threshold_config['threshold']
        predictions = (scores < threshold).astype(int)  # 1 = anomaly, 0 = normal

        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"Anomalies detected: {predictions.sum()} / {len(predictions)} ({100*predictions.mean():.2f}%)")

        return scores, predictions

    def compute_metrics(self, df, scores, predictions):
        """Compute evaluation metrics (dataset-adaptive)"""
        print("\nComputing metrics...")

        metrics = {
            'n_samples': len(scores),
            'n_anomalies': int(predictions.sum()),
            'anomaly_rate': float(predictions.mean()),
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'score_median': float(np.median(scores)),
            'score_q25': float(np.percentile(scores, 25)),
            'score_q75': float(np.percentile(scores, 75)),
            'threshold': self.threshold_config['threshold'],
            'target_far': self.threshold_config.get('target_far_per_week', 'N/A')
        }

        # Check for labels (supervised learning)
        if 'fault_type' in df.columns or 'failure' in df.columns:
            print("Detected labels - computing supervised metrics")

            # Determine label column
            if 'fault_type' in df.columns:
                # CWRU: fault_type == 'normal' is normal, others are anomalies
                y_true = (df['fault_type'] != 'normal').astype(int).values
            else:
                # AI4I: 'failure' column
                y_true = df['failure'].astype(int).values

            # Supervised metrics
            metrics['supervised'] = True
            metrics['precision'] = float(precision_score(y_true, predictions, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, predictions, zero_division=0))
            metrics['f1'] = float(f1_score(y_true, predictions, zero_division=0))
            metrics['accuracy'] = float(accuracy_score(y_true, predictions))

            # ROC and PR curves (need continuous scores)
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, -scores))  # Negative scores for AUC
                metrics['pr_auc'] = float(average_precision_score(y_true, -scores))
            except ValueError:
                print("Warning: Could not compute AUC (may need both classes)")
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
            metrics['confusion_matrix'] = {
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn)
            }

            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1: {metrics['f1']:.3f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}" if metrics['roc_auc'] else "  ROC-AUC: N/A")

        else:
            print("No labels found - computing unsupervised metrics only")
            metrics['supervised'] = False

        return metrics

    def create_visualizations(self, df, scores, predictions, metrics):
        """Generate all visualization plots"""
        print("\nCreating visualizations...")

        plots = {}

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

        try:
            # 1. Score Distribution
            plots['score_distribution'] = self._plot_score_distribution(scores)

            # 2. Timeline Plot (if time-series data)
            if 'timestamp' in df.columns or 'file_index' in df.columns:
                plots['timeline'] = self._plot_timeline(df, scores, predictions)

            # 3. Confusion Matrix (if supervised)
            if metrics.get('supervised'):
                if 'fault_type' in df.columns:
                    y_true = (df['fault_type'] != 'normal').astype(int).values
                else:
                    y_true = df['failure'].astype(int).values
                plots['confusion_matrix'] = self._plot_confusion_matrix(y_true, predictions)

            # 4. ROC Curve (if supervised)
            if metrics.get('supervised') and metrics.get('roc_auc'):
                plots['roc_curve'] = self._plot_roc_curve(y_true, scores)

            # 5. PR Curve (if supervised)
            if metrics.get('supervised') and metrics.get('pr_auc'):
                plots['pr_curve'] = self._plot_pr_curve(y_true, scores)

        except Exception as e:
            print(f"Warning: Error creating visualizations: {e}")

        return plots

    def _plot_score_distribution(self, scores):
        """Plot histogram of anomaly scores"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')

        # Threshold line
        threshold = self.threshold_config['threshold']
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold: {threshold:.4f}')

        # Color regions
        ylim = ax.get_ylim()
        ax.axvspan(scores.min(), threshold, alpha=0.1, color='red', label='Anomaly Region')

        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Anomaly Score Distribution - {self.dataset_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_timeline(self, df, scores, predictions):
        """Plot anomaly scores over time"""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Determine x-axis
        if 'timestamp' in df.columns:
            x = pd.to_datetime(df['timestamp'])
            xlabel = 'Time'
        elif 'file_index' in df.columns:
            x = df['file_index'].values
            xlabel = 'File Index'
        else:
            x = np.arange(len(scores))
            xlabel = 'Sample Index'

        # Plot scores
        ax.plot(x, scores, linewidth=0.5, alpha=0.7, label='Anomaly Score')

        # Threshold line
        threshold = self.threshold_config['threshold']
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')

        # Highlight anomalies
        anomaly_mask = predictions == 1
        if anomaly_mask.any():
            ax.scatter(x[anomaly_mask], scores[anomaly_mask], color='red', s=10,
                      alpha=0.5, label='Detected Anomalies', zorder=5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Anomaly Score', fontsize=12)
        ax.set_title(f'Anomaly Score Timeline - {self.dataset_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_roc_curve(self, y_true, scores):
        """Plot ROC curve"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Compute ROC curve (use negative scores since lower = more anomalous)
        fpr, tpr, _ = roc_curve(y_true, -scores)
        auc = roc_auc_score(y_true, -scores)

        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _plot_pr_curve(self, y_true, scores):
        """Plot Precision-Recall curve"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true, -scores)
        ap = average_precision_score(y_true, -scores)

        ax.plot(recall, precision, linewidth=2, label=f'PR (AP = {ap:.3f})')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_base64

    def generate_shap_explanations(self, X, scores, n_samples=100):
        """Generate SHAP explanations"""
        if not SHAP_AVAILABLE:
            print("\nSHAP not available - skipping explanations")
            return None

        print(f"\nGenerating SHAP explanations (n_samples={n_samples})...")

        try:
            # Limit samples for performance
            n_samples = min(n_samples, len(X))
            X_sample = X[:n_samples]

            # Create explainer
            model_class = type(self.model).__name__
            if 'IsolationForest' in model_class:
                print("Using TreeExplainer (fast)")
                explainer = shap.TreeExplainer(self.model)
            else:
                print("Using KernelExplainer (slower)")
                background = shap.sample(X, min(100, len(X)))
                if hasattr(self.model, 'decision_function'):
                    explainer = shap.KernelExplainer(self.model.decision_function, background)
                else:
                    explainer = shap.KernelExplainer(self.model.score_samples, background)

            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Generate summary plot
            print("Creating SHAP summary plot...")
            shap.summary_plot(shap_values, X_sample,
                            feature_names=self.feature_names,
                            show=False, max_display=15)
            fig = plt.gcf()
            fig.set_size_inches(10, 8)
            plt.tight_layout()
            summary_plot = self._fig_to_base64(fig)

            # Save SHAP values to CSV
            shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
            shap_df.to_csv(self.output_dir / 'shap_values.csv', index=False)
            print(f"Saved SHAP values: {self.output_dir / 'shap_values.csv'}")

            return {
                'summary_plot': summary_plot,
                'shap_values': shap_values
            }

        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            return None

    def generate_html_report(self, metrics, plots, shap_results=None):
        """Generate comprehensive HTML report"""
        print("\nGenerating HTML report...")

        # Get git commit
        try:
            import subprocess
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        except:
            git_hash = 'unknown'

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report - {self.dataset_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric {{
            font-size: 1.1em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .plot {{
            max-width: 100%;
            margin: 20px 0;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .config-block {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Predictive Maintenance Evaluation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Executive Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Dataset</td><td class="metric">{self.dataset_name}</td></tr>
            <tr><td>Model Type</td><td class="metric">{type(self.model).__name__}</td></tr>
            <tr><td>Test Samples</td><td class="metric">{metrics['n_samples']:,}</td></tr>
            <tr><td>Anomalies Detected</td><td class="metric">{metrics['n_anomalies']:,} ({metrics['anomaly_rate']*100:.2f}%)</td></tr>
            <tr><td>Threshold</td><td class="metric">{metrics['threshold']:.6f}</td></tr>
            <tr><td>Target FAR</td><td class="metric">{metrics['target_far']}</td></tr>
        </table>

        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Score Min</td><td>{metrics['score_min']:.4f}</td></tr>
            <tr><td>Score Max</td><td>{metrics['score_max']:.4f}</td></tr>
            <tr><td>Score Mean</td><td>{metrics['score_mean']:.4f}</td></tr>
            <tr><td>Score Std</td><td>{metrics['score_std']:.4f}</td></tr>
            <tr><td>Score Median</td><td>{metrics['score_median']:.4f}</td></tr>
        </table>
        """

        # Add supervised metrics if available
        if metrics.get('supervised'):
            html += f"""
        <h3>Classification Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Precision</td><td class="success">{metrics['precision']:.4f}</td></tr>
            <tr><td>Recall</td><td class="success">{metrics['recall']:.4f}</td></tr>
            <tr><td>F1 Score</td><td class="success">{metrics['f1']:.4f}</td></tr>
            <tr><td>Accuracy</td><td class="success">{metrics['accuracy']:.4f}</td></tr>
            <tr><td>ROC-AUC</td><td class="success">{metrics.get('roc_auc', 'N/A')}</td></tr>
            <tr><td>PR-AUC</td><td class="success">{metrics.get('pr_auc', 'N/A')}</td></tr>
        </table>

        <h3>Confusion Matrix Details</h3>
        <table>
            <tr><th>Metric</th><th>Count</th></tr>
            <tr><td>True Positives (TP)</td><td>{metrics['confusion_matrix']['TP']}</td></tr>
            <tr><td>True Negatives (TN)</td><td>{metrics['confusion_matrix']['TN']}</td></tr>
            <tr><td>False Positives (FP)</td><td class="warning">{metrics['confusion_matrix']['FP']}</td></tr>
            <tr><td>False Negatives (FN)</td><td class="error">{metrics['confusion_matrix']['FN']}</td></tr>
        </table>
            """

        # Add visualizations
        html += "<h2>Visualizations</h2>"
        for plot_name, plot_data in plots.items():
            title = plot_name.replace('_', ' ').title()
            html += f'<h3>{title}</h3><img src="data:image/png;base64,{plot_data}" class="plot"/>'

        # Add SHAP explanations
        if shap_results:
            html += f"""
        <h2>Explainability (SHAP)</h2>
        <p>SHAP (SHapley Additive exPlanations) values show which features contribute most to anomaly detection.</p>
        <h3>Feature Importance</h3>
        <img src="data:image/png;base64,{shap_results['summary_plot']}" class="plot"/>
            """

        # Add configuration
        html += f"""
        <h2>Configuration</h2>
        <h3>Threshold Configuration</h3>
        <div class="config-block">{json.dumps(self.threshold_config, indent=2)}</div>

        <h2>Run Metadata</h2>
        <table>
            <tr><th>Item</th><th>Value</th></tr>
            <tr><td>Timestamp</td><td>{datetime.now().isoformat()}</td></tr>
            <tr><td>Git Commit</td><td>{git_hash}</td></tr>
            <tr><td>Model Path</td><td>{self.model_path}</td></tr>
            <tr><td>Features</td><td>{len(self.feature_names)} features</td></tr>
        </table>
    </div>
</body>
</html>
        """

        # Save HTML report
        report_path = self.output_dir / 'evaluation_report.html'
        with open(report_path, 'w') as f:
            f.write(html)

        print(f"Saved HTML report: {report_path}")
        return report_path

    def run_evaluation(self, split='test', shap_samples=100):
        """Run complete evaluation pipeline"""
        print("="*60)
        print("MODEL EVALUATION - STAGE 5")
        print("="*60)
        print(f"\nDataset: {self.dataset_name}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Split: {split}")

        # Load data
        df = self.load_data(split)

        # Prepare features
        X = self.prepare_features(df)

        # Generate predictions
        scores, predictions = self.predict(X)

        # Compute metrics
        metrics = self.compute_metrics(df, scores, predictions)

        # Create visualizations
        plots = self.create_visualizations(df, scores, predictions, metrics)

        # Generate SHAP explanations
        shap_results = None
        if shap_samples > 0:
            shap_results = self.generate_shap_explanations(X, scores, shap_samples)

        # Save predictions
        predictions_df = df.copy()
        predictions_df['anomaly_score'] = scores
        predictions_df['prediction'] = predictions
        predictions_path = self.output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\nSaved predictions: {predictions_path}")

        # Save metrics
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics: {metrics_path}")

        # Generate HTML report
        self.generate_html_report(metrics, plots, shap_results)

        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"\nReport directory: {self.output_dir}")
        print(f"Open: {self.output_dir / 'evaluation_report.html'}")

        return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained anomaly detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options
    parser.add_argument('--report_dir', help='Report directory containing all artifacts')
    parser.add_argument('--model', help='Path to trained model (.joblib or .pth)')
    parser.add_argument('--threshold', help='Path to threshold config JSON')
    parser.add_argument('--config', help='Dataset config YAML')
    parser.add_argument('--output', help='Output directory (default: same as report_dir)')

    # Evaluation options
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                       help='Data split to evaluate (default: test)')
    parser.add_argument('--shap_samples', type=int, default=100,
                       help='Number of samples for SHAP analysis (0 to disable, default: 100)')

    args = parser.parse_args()

    # Load from report_dir if provided
    if args.report_dir:
        report_dir = Path(args.report_dir)

        # Extract model name from report directory name (e.g., ims_iforest)
        model_name = report_dir.name

        # Models are in artifacts/models/ directory
        models_dir = report_dir.parent.parent / 'models'

        # Find model file matching the name
        model_candidates = list(models_dir.glob(f'{model_name}.*'))
        model_candidates = [f for f in model_candidates if f.suffix in ['.joblib', '.pth']]

        if not model_candidates:
            raise FileNotFoundError(f"No model found in {models_dir} matching '{model_name}'")
        model_path = model_candidates[0]

        # Find threshold config
        threshold_path = report_dir / 'threshold_config.json'
        if not threshold_path.exists():
            raise FileNotFoundError(f"Threshold config not found: {threshold_path}")

        # Load configs
        with open(threshold_path, 'r') as f:
            threshold_config = json.load(f)

        # Find dataset config from run.json
        run_json_path = report_dir / 'run.json'
        if not run_json_path.exists():
            raise FileNotFoundError(f"run.json not found: {run_json_path}")

        with open(run_json_path, 'r') as f:
            run_metadata = json.load(f)

        # Get model config path from run metadata (confusingly named dataset_config)
        model_config_path = run_metadata.get('dataset_config')
        if not model_config_path:
            raise ValueError("dataset_config not found in run.json")

        # Handle relative paths
        if not Path(model_config_path).is_absolute():
            model_config_path = Path(model_config_path)

        # Load model config
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)

        # Get actual dataset config path from model config
        dataset_config_path = model_config.get('dataset_config')
        if not dataset_config_path:
            raise ValueError("dataset_config not found in model config")

        # Load dataset config
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        # Load feature names
        features_file = report_dir / 'features.txt'
        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
        else:
            feature_names = None

        output_dir = report_dir

    else:
        # Use individual arguments
        if not all([args.model, args.threshold, args.config]):
            parser.error("Either --report_dir or all of --model, --threshold, --config are required")

        model_path = Path(args.model)
        threshold_path = Path(args.threshold)

        with open(threshold_path, 'r') as f:
            threshold_config = json.load(f)

        with open(args.config, 'r') as f:
            dataset_config = yaml.safe_load(f)

        feature_names = None
        output_dir = Path(args.output) if args.output else model_path.parent.parent / 'reports' / model_path.stem

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        threshold_config=threshold_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        feature_names=feature_names
    )

    # Run evaluation
    evaluator.run_evaluation(split=args.split, shap_samples=args.shap_samples)


if __name__ == '__main__':
    main()
