"""Unit tests for convex polyhedral invariant constraint set auditor."""

import numpy as np
import pytest

from kinesis.safety.polyhedral import PolyhedralConstraintSet, PolyhedralAuditResult


def test_polyhedral_box_constraints():
    """Verify box constraints [-1, 1] x [-2, 2] represented in H * x <= h format."""
    # -1 <= x0 <= 1 -> x0 <= 1, -x0 <= 1
    # -2 <= x1 <= 2 -> x1 <= 2, -x1 <= 2
    H = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ], dtype=np.float64)
    h = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)

    poly = PolyhedralConstraintSet(H, h)

    # Interior point
    x_in = np.array([0.0, 0.5])
    assert poly.contains(x_in) is True
    audit_in = poly.audit(x_in)
    assert audit_in.is_contained is True
    assert audit_in.euclidean_safety_margin > 0.0
    np.testing.assert_allclose(audit_in.projected_safe_state, x_in, atol=1e-6)

    # Exterior point
    x_out = np.array([1.5, 0.0])
    assert poly.contains(x_out) is False
    audit_out = poly.audit(x_out)
    assert audit_out.is_contained is False
    assert audit_out.euclidean_safety_margin < 0.0
    assert audit_out.most_critical_constraint_idx == 0

    # Projection should bring x0 to 1.0
    projected = audit_out.projected_safe_state
    assert poly.contains(projected) is True
    np.testing.assert_allclose(projected, np.array([1.0, 0.0]), atol=1e-5)


def test_polyhedral_halfspace_simplex():
    """Verify simplex constraint x0 + x1 <= 1, x0 >= 0, x1 >= 0."""
    H = np.array([
        [1.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float64)
    h = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    poly = PolyhedralConstraintSet(H, h)

    assert poly.contains(np.array([0.2, 0.3])) is True
    assert poly.contains(np.array([0.8, 0.8])) is False

    # Projection of [1.0, 1.0] onto x0 + x1 <= 1
    proj = poly.project_onto_set(np.array([1.0, 1.0]))
    assert poly.contains(proj) is True
    np.testing.assert_allclose(proj, np.array([0.5, 0.5]), atol=1e-4)
