"""Unit tests for discrete-time Sliding Mode Controller."""

import numpy as np
import pytest

from kinesis.analytic.sliding_mode import DiscreteSlidingModeController


def test_smc_regulation_under_matched_disturbance():
    """Verify sliding mode controller drives sliding surface to zero under parameter shift."""
    dt = 0.05
    A_nom = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    B_nom = np.array([[0.0], [dt]], dtype=np.float64)
    C_surf = np.array([[2.0, 1.0]], dtype=np.float64)

    smc = DiscreteSlidingModeController(
        A_nominal=A_nom,
        B_nominal=B_nom,
        sliding_matrix_C=C_surf,
        switching_gain=1.0,
        boundary_layer_thickness=0.05,
        action_bounds=(-2.0, 2.0),
    )

    # Actual system has +20% mass disturbance (actuator attenuation)
    B_perturbed = B_nom * 0.8
    x = np.array([1.0, -0.5], dtype=np.float64)

    for step in range(40):
        res = smc.compute_action(x)
        assert np.isfinite(res.action[0])
        # Step perturbed system with matched disturbance
        disturbance = 0.1 * np.sin(step * 0.2)
        x = A_nom @ x + B_perturbed @ res.action + B_perturbed @ np.array([disturbance])

    final_res = smc.compute_action(x)
    # Sliding surface should be close to boundary layer
    assert np.abs(final_res.sliding_surface_value[0]) < 0.2


def test_smc_boundary_layer_saturation():
    """Verify that within boundary layer, switching control is proportional, not discontinuous."""
    A_nom = np.eye(2)
    B_nom = np.array([[0.0], [1.0]])
    C_surf = np.array([[1.0, 1.0]])

    smc = DiscreteSlidingModeController(
        A_nominal=A_nom,
        B_nominal=B_nom,
        sliding_matrix_C=C_surf,
        switching_gain=0.8,
        boundary_layer_thickness=0.1,
    )

    x_inside = np.array([0.02, 0.02])
    res_inside = smc.compute_action(x_inside)
    assert res_inside.in_boundary_layer is True
    # Switching action should be scaled down by (s / epsilon)
    assert np.abs(res_inside.switching_action[0]) < 0.8
