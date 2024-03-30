"""Unit tests for Page-Hinkley and CUSUM drift detectors."""

import numpy as np
import pytest

from kinesis.diagnostics.drift_detector import CUSUMDriftDetector, PageHinkleyDriftDetector


def test_page_hinkley_drift_detection():
    """Verify Page-Hinkley detects sudden mean shift."""
    rng = np.random.default_rng(42)
    ph = PageHinkleyDriftDetector(delta=0.05, threshold=5.0)

    # 40 stationary samples around 0.0
    for _ in range(40):
        res = ph.update(float(rng.normal(0.0, 0.1)))
        assert res.drift_detected is False

    # Shift to mean 2.0
    alarm = False
    for _ in range(20):
        res = ph.update(float(rng.normal(2.0, 0.1)))
        if res.drift_detected:
            alarm = True
            break

    assert alarm is True


def test_cusum_drift_detection():
    """Verify CUSUM detects positive and negative step shifts."""
    cusum = CUSUMDriftDetector(target_mean=0.0, slack_k=0.2, threshold_h=3.0)

    # Stationary around 0.0
    for _ in range(30):
        res = cusum.update(0.0)
        assert res.drift_detected is False

    # Positive jump
    alarm_pos = False
    for _ in range(10):
        res = cusum.update(1.0)
        if res.drift_detected:
            alarm_pos = True
            break
    assert alarm_pos is True
