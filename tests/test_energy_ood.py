"""Unit tests for Energy-based OOD detector."""

import numpy as np
import pytest
import torch

from kinesis.uncertainty.energy_ood import EnergyOODDetector


def test_energy_ood_detection():
    """Verify that anomalous inputs have higher free energy than in-distribution data."""
    detector = EnergyOODDetector(temperature=1.0, percentile_threshold=95.0)

    # In-distribution logits: strong peak, high likelihood -> lower free energy
    in_dist = torch.tensor([[5.0, 0.1, 0.2], [4.5, 0.2, 0.1], [5.2, 0.0, 0.1]], dtype=torch.float32)
    threshold = detector.fit(in_dist)

    # In-distribution query
    in_query = torch.tensor([[4.8, 0.1, 0.1]], dtype=torch.float32)
    res_in = detector.detect(in_query)
    assert res_in.is_ood is False

    # Out-of-distribution logits: flat, low likelihood -> high free energy
    ood_query = torch.tensor([[-5.0, -4.8, -5.2]], dtype=torch.float32)
    res_ood = detector.detect(ood_query)
    assert res_ood.energy_score > threshold
    assert res_ood.is_ood is True
    assert res_ood.anomaly_probability > 0.8
