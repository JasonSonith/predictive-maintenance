"""
Unit tests for make_features.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from make_features import (
    compute_time_domain_features,
    create_windows,
    validate_features
)


class TestComputeFeatures:
    """Test time-domain feature computation"""

    def test_basic_statistics(self):
        """Test mean, std, min, max"""
        data = np.array([1, 2, 3, 4, 5])
        features = compute_time_domain_features(data, ['mean', 'std', 'min', 'max'])

        assert features['mean'] == 3.0
        assert features['std'] == pytest.approx(np.std([1, 2, 3, 4, 5]))
        assert features['min'] == 1.0
        assert features['max'] == 5.0

    def test_rms(self):
        """Test RMS computation"""
        data = np.array([1, 2, 3, 4, 5])
        features = compute_time_domain_features(data, ['rms'])

        expected_rms = np.sqrt(np.mean(data**2))
        assert features['rms'] == pytest.approx(expected_rms)

    def test_peak_to_peak(self):
        """Test peak-to-peak amplitude"""
        data = np.array([1, 2, 3, 4, 5])
        features = compute_time_domain_features(data, ['peak_to_peak'])

        assert features['peak_to_peak'] == 4.0  # 5 - 1

    def test_kurtosis_skewness(self):
        """Test distribution shape features"""
        # Normal distribution should have kurtosis ~0 (excess)
        np.random.seed(42)
        data = np.random.randn(1000)

        features = compute_time_domain_features(data, ['kurtosis', 'skewness'])

        # Should be close to 0 for normal distribution
        assert abs(features['kurtosis']) < 0.5
        assert abs(features['skewness']) < 0.5

    def test_crest_factor(self):
        """Test crest factor computation"""
        data = np.array([1, 1, 1, 10, 1, 1, 1])  # One large peak
        features = compute_time_domain_features(data, ['crest_factor'])

        rms = np.sqrt(np.mean(data**2))
        expected_cf = 10 / rms
        assert features['crest_factor'] == pytest.approx(expected_cf)

    def test_empty_data(self):
        """Test handling of empty data"""
        data = np.array([])
        features = compute_time_domain_features(data, ['mean', 'std', 'rms'])

        assert np.isnan(features['mean'])
        assert np.isnan(features['std'])
        assert np.isnan(features['rms'])

    def test_nan_data(self):
        """Test handling of NaN values"""
        data = np.array([1, 2, np.nan, 4, 5])
        features = compute_time_domain_features(data, ['mean', 'rms'])

        # Should compute on non-NaN values
        assert features['mean'] == 3.0  # mean of [1,2,4,5]

    def test_pandas_series_input(self):
        """Test that function handles pandas Series"""
        data = pd.Series([1, 2, 3, 4, 5])
        features = compute_time_domain_features(data, ['mean', 'std'])

        assert features['mean'] == 3.0


class TestCreateWindows:
    """Test window creation"""

    def test_non_overlapping_windows(self):
        """Test non-overlapping windows"""
        data = np.arange(10)  # [0, 1, 2, ..., 9]
        windows = create_windows(data, window_size=5, stride=5)

        assert len(windows) == 2
        assert np.array_equal(windows[0], [0, 1, 2, 3, 4])
        assert np.array_equal(windows[1], [5, 6, 7, 8, 9])

    def test_overlapping_windows(self):
        """Test overlapping windows (50% overlap)"""
        data = np.arange(10)
        windows = create_windows(data, window_size=5, stride=2)

        # Should create windows at positions 0, 2, 4
        assert len(windows) == 3
        assert np.array_equal(windows[0], [0, 1, 2, 3, 4])
        assert np.array_equal(windows[1], [2, 3, 4, 5, 6])
        assert np.array_equal(windows[2], [4, 5, 6, 7, 8])

    def test_single_window(self):
        """Test when data fits exactly one window"""
        data = np.arange(5)
        windows = create_windows(data, window_size=5, stride=1)

        assert len(windows) == 1
        assert np.array_equal(windows[0], data)

    def test_data_too_short(self):
        """Test when data is shorter than window size"""
        data = np.arange(3)
        windows = create_windows(data, window_size=5, stride=1)

        assert len(windows) == 0

    def test_pandas_series_input(self):
        """Test that function handles pandas Series"""
        data = pd.Series(np.arange(10))
        windows = create_windows(data, window_size=5, stride=5)

        assert len(windows) == 2
        assert isinstance(windows[0], np.ndarray)


class TestValidateFeatures:
    """Test feature validation"""

    def test_valid_features(self):
        """Test validation with valid features"""
        df = pd.DataFrame({
            'mean': [1.0, 2.0, 3.0],
            'std': [0.5, 0.6, 0.7],
            'rms': [1.1, 2.1, 3.1]
        })

        config = {'schema': {'computed_features': ['mean', 'std', 'rms']}}
        result = validate_features(df, config)

        assert result.equals(df)

    def test_nan_handling(self):
        """Test that NaN values are filled"""
        df = pd.DataFrame({
            'mean': [1.0, np.nan, 3.0],
            'std': [0.5, 0.6, np.nan]
        })

        config = {'schema': {'computed_features': ['mean', 'std']}}
        result = validate_features(df, config)

        # NaN values should be filled with 0.0
        assert result['mean'].isna().sum() == 0
        assert result['std'].isna().sum() == 0
        assert result.loc[1, 'mean'] == 0.0
        assert result.loc[2, 'std'] == 0.0


class TestIntegration:
    """Integration tests for the full pipeline"""

    def test_simple_feature_extraction(self):
        """Test end-to-end feature extraction on synthetic data"""
        # Create synthetic signal
        signal = np.sin(np.linspace(0, 10*np.pi, 1000))

        # Create windows
        windows = create_windows(signal, window_size=100, stride=50)

        assert len(windows) > 0

        # Compute features for first window
        features = compute_time_domain_features(
            windows[0],
            ['mean', 'std', 'rms', 'kurtosis', 'skewness']
        )

        # Verify all features computed
        assert 'mean' in features
        assert 'std' in features
        assert 'rms' in features
        assert 'kurtosis' in features
        assert 'skewness' in features

        # Verify no NaN
        assert all(not np.isnan(v) for v in features.values())


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_constant_signal(self):
        """Test features on constant signal"""
        data = np.ones(100)
        features = compute_time_domain_features(
            data,
            ['mean', 'std', 'rms', 'peak_to_peak']
        )

        assert features['mean'] == 1.0
        assert features['std'] == 0.0
        assert features['rms'] == 1.0
        assert features['peak_to_peak'] == 0.0

    def test_zero_signal(self):
        """Test features on zero signal"""
        data = np.zeros(100)
        features = compute_time_domain_features(data, ['mean', 'rms', 'crest_factor'])

        assert features['mean'] == 0.0
        assert features['rms'] == 0.0
        assert features['crest_factor'] == 0.0  # Handles divide by zero

    def test_very_small_window(self):
        """Test with minimum viable window size"""
        data = np.array([1, 2, 3, 4])
        features = compute_time_domain_features(
            data,
            ['mean', 'std', 'kurtosis', 'skewness']
        )

        # Kurtosis requires 4+ samples, skewness requires 3+
        assert not np.isnan(features['mean'])
        assert not np.isnan(features['std'])
        assert not np.isnan(features['kurtosis'])
        assert not np.isnan(features['skewness'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
