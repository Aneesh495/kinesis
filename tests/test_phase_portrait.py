"""Unit tests for Phase Portrait trajectory analyzer."""

import numpy as np
import pytest

from kinesis.diagnostics.phase_portrait import PhasePortraitAnalyzer


def test_phase_portrait_circle_area():
    """Verify that circular orbit in phase space computes area approx pi * r^2."""
    analyzer = PhasePortraitAnalyzer(q_bounds=(-2.0, 2.0), v_bounds=(-2.0, 2.0))

    r = 1.0
    thetas = np.linspace(0, 2 * np.pi, 200)
    q = r * np.cos(thetas)
    v = r * np.sin(thetas)

    metrics = analyzer.compute_metrics(q, v)
    expected_area = np.pi * r**2
    assert np.isclose(metrics.orbit_enclosed_area, expected_area, atol=0.05)
    assert metrics.apparent_limit_cycle is True


def test_ascii_phase_portrait_rendering():
    """Verify ASCII grid contains start and end marker symbols."""
    analyzer = PhasePortraitAnalyzer(q_bounds=(-1.0, 1.0), v_bounds=(-1.0, 1.0))
    q = np.array([-0.5, 0.0, 0.5])
    v = np.array([-0.5, 0.0, 0.5])

    art = analyzer.render_ascii_portrait(q, v, width=20, height=10)
    assert "S" in art
    assert "E" in art
    assert "+" in art
