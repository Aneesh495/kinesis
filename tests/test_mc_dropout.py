"""Unit tests for Monte Carlo Dropout epistemic uncertainty estimator."""

import pytest
import torch
import torch.nn as nn

from kinesis.uncertainty.mc_dropout import MCDropoutEstimator


def test_mc_dropout_variance():
    """Verify that MC Dropout produces non-zero epistemic variance."""
    model = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(32, 2),
    )

    estimator = MCDropoutEstimator(model, num_samples=15)
    x = torch.randn(4, 8)

    res = estimator.estimate(x)
    assert res.mean.shape == (4, 2)
    assert res.epistemic_variance.shape == (4, 2)
    assert res.samples.shape == (15, 4, 2)

    # Variance should be strictly positive due to dropout masking
    assert torch.all(res.epistemic_variance > 0.0)


def test_mc_dropout_deterministic_when_no_dropout():
    """Verify that a model without dropout yields zero variance."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    estimator = MCDropoutEstimator(model, num_samples=5)
    x = torch.randn(2, 4)

    res = estimator.estimate(x)
    assert torch.allclose(res.epistemic_variance, torch.zeros_like(res.epistemic_variance), atol=1e-6)
