"""Unit tests for discrete Controllability and Observability Gramian analysis."""

import numpy as np
import pytest

from kinesis.analytic.gramian import DiscreteGramianAnalyzer, GramianAnalysisResult


def test_controllable_and_observable_system():
    """Verify Gramian analysis on a canonical observable and controllable system."""
    A = np.array([[0.8, 0.2], [0.0, 0.6]], dtype=np.float64)
    B = np.array([[0.0], [1.0]], dtype=np.float64)
    C = np.array([[1.0, 0.0]], dtype=np.float64)

    analyzer = DiscreteGramianAnalyzer(A, B, C)
    res = analyzer.analyze()

    assert isinstance(res, GramianAnalysisResult)
    assert res.is_controllable is True
    assert res.is_observable is True
    assert res.controllability_rank == 2
    assert res.observability_rank == 2
    assert res.min_controllability_singular_value > 0.0
    assert res.min_observability_singular_value > 0.0

    # Test finite horizon
    res_finite = analyzer.analyze(horizon=20)
    assert res_finite.is_controllable is True
    assert res_finite.is_observable is True


def test_uncontrollable_system():
    """Verify that uncontrollable modes are identified by rank deficit."""
    A = np.array([[0.5, 0.0], [0.0, 0.4]], dtype=np.float64)
    B = np.array([[1.0], [0.0]], dtype=np.float64)  # State 2 cannot be controlled
    C = np.array([[1.0, 1.0]], dtype=np.float64)

    analyzer = DiscreteGramianAnalyzer(A, B, C)
    res = analyzer.analyze()

    assert res.is_controllable is False
    assert res.controllability_rank == 1
    assert res.is_observable is True
    assert res.observability_rank == 2
