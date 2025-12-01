#!/usr/bin/env python3
"""
Generate all presentation figures for the project.

This script creates publication-ready figures for slides 4, 5, 8, 9, and 10
using real data from the trained models and datasets.

Usage:
    python scripts/generate_presentation_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter1d

# Paths
OUTPUT_DIR = Path("figures/presentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection points for slide 09 (from threshold analysis)
DETECTION_POINTS = {
    'ims_iforest': 1660,
    'ims_autoencoder': 1467,
    'ims_knn_lof': 1467,
    'ims_ocsvm': 1670
}

# Model configurations for slide 09
MODELS = {
    'ims_iforest': {
        'name': 'Isolation Forest',
        'color': '#2E86AB',
        'scores_file': 'ims_iforest_scores.csv',
        'threshold_config': 'ims_iforest/threshold_config.json'
    },
    'ims_autoencoder': {
        'name': 'AutoEncoder',
        'color': '#E63946',
        'scores_file': 'ims_autoencoder_scores.csv',
        'threshold_config': 'ims_autoencoder/threshold_config.json'
    },
    'ims_knn_lof': {
        'name': 'kNN-LOF',
        'color': '#06A77D',
        'scores_file': 'ims_knn_lof_scores.csv',
        'threshold_config': 'ims_knn_lof/threshold_config.json'
    },
    'ims_ocsvm': {
        'name': 'One-Class SVM',
        'color': '#9D4EDD',
        'scores_file': 'ims_ocsvm_scores.csv',
        'threshold_config': 'ims_ocsvm/threshold_config.json'
    }
}


def generate_slide_04_raw_waveform():
    """Generate raw vibration waveform from first IMS file"""
    print("\n" + "="*70)
    print("Generating Slide 04: Raw Waveform")
    print("="*70)

    raw_file = Path("data/raw/ims/1st_test/2003.10.22.12.06.24")
    df = pd.read_csv(raw_file, sep='\t', header=None,
                     names=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'])

    signal = df['ch1'].iloc[:2048].values
    time = np.arange(len(signal))

    print(f"Loaded {len(signal)} samples from {raw_file.name}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, signal, color='#2E86AB', linewidth=1, alpha=0.8)
    ax.set_xlabel('Sample Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Amplitude (Vibration)', fontsize=13, fontweight='bold')
    ax.set_title('Raw Bearing Vibration Signal (IMS Dataset)\nChannel 1, First File (Healthy Bearing)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(signal))
    ax.text(0.02, 0.98, f'Sample Rate: 20 kHz\nDuration: {len(signal)/20000:.3f} seconds\nWindow Size: 2048 samples',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "slide_04_raw_waveform.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_slide_05_sliding_window():
    """Generate sliding window visualization"""
    print("\n" + "="*70)
    print("Generating Slide 05: Sliding Window")
    print("="*70)

    raw_file = Path("data/raw/ims/1st_test/2003.10.22.12.06.24")
    df = pd.read_csv(raw_file, sep='\t', header=None,
                     names=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'])

    signal = df['ch1'].iloc[:6000].values
    time = np.arange(len(signal))
    window_size = 2048
    stride = 1024

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(time, signal, color='#2E86AB', linewidth=1, alpha=0.6, label='Raw Signal')

    colors = ['#E63946', '#06A77D', '#9D4EDD', '#FFB703']
    window_starts = [0, stride, stride*2, stride*3]

    for i, (start, color) in enumerate(zip(window_starts, colors)):
        end = start + window_size
        if end <= len(signal):
            ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)
            ax.text(start + window_size/2, signal.max() * 0.9, f'Window {i+1}',
                   fontsize=11, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                            alpha=0.7, edgecolor='black', linewidth=1.5))

    signal_range = signal.max() - signal.min()
    y_arrow = signal.min() - signal_range * 0.15

    for i in range(3):
        start = window_starts[i]
        end = window_starts[i+1]
        ax.annotate('', xy=(end, y_arrow), xytext=(start, y_arrow),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text((start + end)/2, y_arrow - signal_range * 0.06,
               f'Stride\n{stride}', fontsize=9, ha='center', fontweight='bold')

    ax.set_xlabel('Sample Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Amplitude (Vibration)', fontsize=13, fontweight='bold')
    ax.set_title('Sliding Window Feature Extraction (50% Overlap)\nWindow Size: 2048 samples, Stride: 1024 samples',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, window_starts[3] + window_size + 200)
    ax.set_ylim(signal.min() - signal_range * 0.25, signal.max() * 1.05)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "slide_05_sliding_window.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_slide_08_threshold_distribution():
    """Generate anomaly score distribution with threshold"""
    print("\n" + "="*70)
    print("Generating Slide 08: Threshold Distribution")
    print("="*70)

    df = pd.read_csv("artifacts/scores/ims_iforest_scores.csv")

    with open("artifacts/reports/ims_iforest/threshold_config.json") as f:
        config = json.load(f)

    threshold = config['threshold']
    target_far = config['target_far_per_week']
    achieved_far = config['estimated_far_per_week']

    features_df = pd.read_parquet("data/features/ims/ims_features.parquet")
    n_files = features_df['file_index'].nunique()
    val_start = int(n_files * 0.60)
    val_end = int(n_files * 0.70)
    val_files = features_df[
        (features_df['file_index'] >= val_start) &
        (features_df['file_index'] < val_end)
    ]['file_index'].unique()
    val_scores = df[df['file_index'].isin(val_files)]['anomaly_score'].values

    print(f"Validation set: {len(val_scores)} samples")
    print(f"Raw score range: [{val_scores.min():.4f}, {val_scores.max():.4f}]")
    print(f"Raw threshold: {threshold:.4f}")

    # INVERT SCORES: sklearn Isolation Forest outputs negative scores where more negative = more anomalous
    # We flip them so HIGHER = MORE ANOMALOUS (intuitive!)
    val_scores_inverted = -val_scores
    threshold_inverted = -threshold

    print(f"Inverted score range: [{val_scores_inverted.min():.4f}, {val_scores_inverted.max():.4f}]")
    print(f"Inverted threshold: {threshold_inverted:.4f} (higher values = more anomalous)")

    fig, ax = plt.subplots(figsize=(12, 6))

    bins = 50
    counts, edges, patches = ax.hist(val_scores_inverted, bins=bins, color='#2E86AB',
                                      alpha=0.7, edgecolor='black', linewidth=0.5)

    # Color bars: scores ABOVE threshold are anomalous (now intuitive!)
    for patch, edge in zip(patches, edges[:-1]):
        if edge > threshold_inverted:  # Changed from < to >
            patch.set_facecolor('#FFB3B3')  # Red for anomalous
        else:
            patch.set_facecolor('#E8F4F8')  # Blue for normal

    ax.axvline(threshold_inverted, color='black', linestyle='--', linewidth=3,
               label=f'Threshold = {threshold_inverted:.3f}', zorder=10)
    ax.axvspan(val_scores_inverted.min(), threshold_inverted, alpha=0.1, color='lightblue',
               label='Normal Region', zorder=0)
    ax.axvspan(threshold_inverted, val_scores_inverted.max(), alpha=0.1, color='lightcoral',
               label='Anomaly Region', zorder=0)

    ax.set_xlabel('Anomaly Score (Higher = More Anomalous)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Anomaly Score Distribution with Calibrated Threshold\nIsolation Forest on IMS Validation Set',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, bbox_to_anchor=(0.01, 0.99))
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "slide_08_threshold_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_slide_09_ims_timeseries():
    """Generate IMS timeseries with all 4 models and detection markers"""
    print("\n" + "="*70)
    print("Generating Slide 09: IMS Timeseries")
    print("="*70)

    def load_and_process(model_key):
        config = MODELS[model_key]
        df = pd.read_csv(f"artifacts/scores/{config['scores_file']}")

        with open(f"artifacts/reports/{config['threshold_config']}") as f:
            threshold_config = json.load(f)

        file_scores = df.groupby('file_index')['anomaly_score'].mean().reset_index()
        file_scores = file_scores.sort_values('file_index')
        return file_scores, threshold_config

    def normalize_robust(scores):
        p1 = np.percentile(scores, 1)
        p99 = np.percentile(scores, 99)
        scores_clipped = np.clip(scores, p1, p99)
        anomaly_index = 100 * (1 - (scores_clipped - p1) / (p99 - p1))
        return anomaly_index

    model_data = {}
    for model_key in MODELS.keys():
        file_scores, threshold_config = load_and_process(model_key)
        anomaly_index = normalize_robust(file_scores['anomaly_score'].values)
        anomaly_smooth = gaussian_filter1d(anomaly_index, sigma=20)

        detection_file = DETECTION_POINTS.get(model_key)
        detection_idx = np.where(file_scores['file_index'].values == detection_file)[0]
        detection_score = anomaly_smooth[detection_idx[0]] if len(detection_idx) > 0 else None

        model_data[model_key] = {
            'file_indices': file_scores['file_index'].values,
            'anomaly_index': anomaly_smooth,
            'target_far': threshold_config['target_far_per_week'],
            'achieved_far': threshold_config['estimated_far_per_week'],
            'detection_file': detection_file,
            'detection_score': detection_score
        }

    fig, ax = plt.subplots(figsize=(14, 7))

    for model_key, data in model_data.items():
        config = MODELS[model_key]
        ax.plot(data['file_indices'], data['anomaly_index'],
                label=f"{config['name']} (FAR: {data['achieved_far']:.3f}/week)",
                color=config['color'], linewidth=2.5, alpha=0.9, zorder=5)

        if data['detection_file'] and data['detection_score']:
            ax.scatter(data['detection_file'], data['detection_score'],
                      s=200, color=config['color'], marker='o',
                      edgecolors='black', linewidths=2, zorder=10)
            ax.plot([data['detection_file'], data['detection_file']],
                   [0, data['detection_score']],
                   color=config['color'], linestyle=':', linewidth=1.5, alpha=0.5, zorder=3)

    earliest_detection = min(DETECTION_POINTS.values())
    ax.axvspan(0, 500, alpha=0.1, color='lightgreen', zorder=0)
    ax.text(250, 8, 'Healthy\nBaseline', fontsize=11, ha='center',
            fontweight='bold', color='darkgreen')
    ax.text(1100, 8, 'Stable Operation', fontsize=11, ha='center',
            style='italic', color='gray')
    ax.axvspan(earliest_detection, 2156, alpha=0.15, color='orange', zorder=0)
    ax.text(1800, 92, 'Degradation\nDetected', fontsize=11, ha='center',
            fontweight='bold', color='darkorange',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='orange',
                      alpha=0.8, edgecolor='darkorange', linewidth=2))
    ax.axvline(x=2156, color='darkred', linestyle='-', linewidth=4,
               alpha=0.9, zorder=15, label='Bearing Failure')

    ax.set_xlabel('File Number (Chronological Time →)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Anomaly Index (0=Normal, 100=Anomalous)', fontsize=14, fontweight='bold')
    ax.set_title('IMS Bearing Run-to-Failure: Multi-Model Degradation Detection\nMarkers Show First Detection of Degradation',
                 fontsize=15, fontweight='bold', pad=20)

    handles, labels = ax.get_legend_handles_labels()
    legend_items = [(h, l) for h, l in zip(handles, labels) if 'Detection' not in l]
    ax.legend([h for h, l in legend_items], [l for h, l in legend_items],
              loc='upper left', fontsize=10, framealpha=0.95, bbox_to_anchor=(0.01, 0.99))
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xlim(-50, 2250)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    plt.tight_layout()
    output_path = OUTPUT_DIR / "slide_09_ims_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_slide_10_far_comparison():
    """Generate FAR comparison bar chart"""
    print("\n" + "="*70)
    print("Generating Slide 10: FAR Comparison")
    print("="*70)

    threshold_files = [
        ("IMS\nIForest", "ims_iforest"),
        ("IMS\nAutoEncoder", "ims_autoencoder"),
        ("IMS\nkNN-LOF", "ims_knn_lof"),
        ("IMS\nOne-Class\nSVM", "ims_ocsvm"),
        ("AI4I\nIForest", "ai4i_iforest"),
        ("CWRU\nIForest", "cwru_iforest"),
        ("C-MAPSS\nFD001", "fd001_iforest"),
        ("C-MAPSS\nFD002", "fd002_iforest"),
        ("C-MAPSS\nFD003", "fd003_iforest"),
        ("C-MAPSS\nFD004", "fd004_iforest"),
    ]

    models_data = []
    for label, model_key in threshold_files:
        with open(f"artifacts/reports/{model_key}/threshold_config.json") as f:
            config = json.load(f)
        models_data.append({
            'label': label,
            'target': config['target_far_per_week'],
            'achieved': config['estimated_far_per_week']
        })

    fig, ax = plt.subplots(figsize=(14, 7))

    labels = [d['label'] for d in models_data]
    target_vals = [d['target'] for d in models_data]
    achieved_vals = [d['achieved'] for d in models_data]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, target_vals, width, label='Target FAR',
                   color='#A7C6DA', edgecolor='black', linewidth=1.5, alpha=0.9)
    bars2 = ax.bar(x + width/2, achieved_vals, width, label='Achieved FAR',
                   color='#2E86AB', edgecolor='black', linewidth=1.5, alpha=0.9)

    for i, (target, achieved, bar) in enumerate(zip(target_vals, achieved_vals, bars2)):
        accuracy = 1 - abs(target - achieved) / target
        if accuracy >= 0.95:
            bar.set_facecolor('#06A77D')
        elif accuracy >= 0.85:
            bar.set_facecolor('#FFB703')
        else:
            bar.set_facecolor('#E63946')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Dataset & Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('False Alarm Rate (alarms/week)', fontsize=13, fontweight='bold')
    ax.set_title('FAR Calibration Performance Across All Datasets\n10 Models Successfully Calibrated',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "slide_10_far_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("GENERATING ALL PRESENTATION FIGURES")
    print("="*70)

    generate_slide_04_raw_waveform()
    generate_slide_05_sliding_window()
    generate_slide_08_threshold_distribution()
    generate_slide_09_ims_timeseries()
    generate_slide_10_far_comparison()

    print("\n" + "="*70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("slide_*.png")):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
