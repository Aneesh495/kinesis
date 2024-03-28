"""Unit tests for Actuator Spectral Analyzer."""

import numpy as np
import pytest

from kinesis.diagnostics.spectral_analysis import ActuatorSpectralAnalyzer


def test_spectral_pure_sine_detection():
    """Verify that pure low-frequency sinusoid has dominant peak at input frequency."""
    fs = 40.0
    analyzer = ActuatorSpectralAnalyzer(sampling_rate_hz=fs, high_freq_cutoff_hz=10.0)

    # 2 Hz pure sinusoidal signal
    t = np.arange(0, 2.0, 1.0 / fs)
    signal = np.sin(2.0 * np.pi * 2.0 * t)

    res = analyzer.analyze(signal)
    assert np.isclose(res.dominant_frequency_hz, 2.0, atol=0.5)
    assert res.high_frequency_chatter_ratio < 0.1
    assert res.is_excessive_chatter is False


def test_spectral_high_frequency_chatter_flag():
    """Verify that high-frequency switching chatter is flagged."""
    fs = 40.0
    analyzer = ActuatorSpectralAnalyzer(
        sampling_rate_hz=fs,
        high_freq_cutoff_hz=8.0,
        chatter_ratio_threshold=0.3,
    )

    # 15 Hz high-frequency switching chatter
    t = np.arange(0, 2.0, 1.0 / fs)
    signal = np.sign(np.sin(2.0 * np.pi * 15.0 * t))

    res = analyzer.analyze(signal)
    assert res.high_frequency_chatter_ratio > 0.5
    assert res.is_excessive_chatter is True
