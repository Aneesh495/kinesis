"""Unit tests for Conformal Prediction intervals."""

import numpy as np
import pytest

from kinesis.uncertainty.conformal import ConformalPredictor


def test_conformal_coverage_guarantee():
    """Verify that conformal intervals achieve requested 90% marginal coverage."""
    rng = np.random.default_rng(42)
    n_cal = 500
    n_test = 500

    # Synthetic predictions with heteroscedastic noise
    cal_preds = rng.normal(0.0, 1.0, size=(n_cal, 2))
    cal_noise = rng.normal(0.0, 0.5, size=(n_cal, 2))
    cal_targets = cal_preds + cal_noise

    predictor = ConformalPredictor(significance_level=0.1)  # 90% coverage
    q = predictor.calibrate(cal_preds, cal_targets)
    assert q > 0.0

    test_preds = rng.normal(0.0, 1.0, size=(n_test, 2))
    test_noise = rng.normal(0.0, 0.5, size=(n_test, 2))
    test_targets = test_preds + test_noise

    emp_coverage = predictor.evaluate_coverage(test_preds, test_targets)
    # Empirical coverage should be approximately 90% (>= 85%)
    assert emp_coverage >= 0.85


def test_conformal_uncalibrated_error():
    """Verify that predicting without calibration raises an assertion."""
    predictor = ConformalPredictor(significance_level=0.05)
    with pytest.raises(AssertionError):
        predictor.predict_interval(np.array([1.0, 2.0]))
