"""Unit tests for MPC temporal warm-start cache."""

import numpy as np
import pytest

from kinesis.planning.warm_start import WarmStartCache, WarmStartConfig


def test_warm_start_shift_repeat_last():
    """Verify 1-step shift with repeat_last terminal action padding."""
    config = WarmStartConfig(
        horizon=4,
        action_dim=2,
        shift_steps=1,
        terminal_mode="repeat_last",
    )
    cache = WarmStartCache(config)

    # Initial plan: 4 steps
    plan = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ], dtype=np.float64)

    cache.update(plan, cost=10.5)
    assert cache.is_active is True

    warm = cache.get_warm_start()
    assert warm is not None
    assert warm.shape == (4, 2)

    # Shifted: [2, 2], [3, 3], [4, 4], [4, 4]
    np.testing.assert_allclose(warm[0], np.array([2.0, 2.0]))
    np.testing.assert_allclose(warm[1], np.array([3.0, 3.0]))
    np.testing.assert_allclose(warm[2], np.array([4.0, 4.0]))
    np.testing.assert_allclose(warm[3], np.array([4.0, 4.0]))


def test_warm_start_zero_padding():
    """Verify zero padding mode on shift."""
    config = WarmStartConfig(
        horizon=3,
        action_dim=1,
        shift_steps=1,
        terminal_mode="zero",
    )
    cache = WarmStartCache(config)

    plan = np.array([[5.0], [6.0], [7.0]], dtype=np.float64)
    cache.update(plan)

    warm = cache.get_warm_start()
    assert warm is not None
    np.testing.assert_allclose(warm, np.array([[6.0], [7.0], [0.0]]))


def test_warm_start_invalidation():
    """Verify divergence triggers cache invalidation."""
    config = WarmStartConfig(
        horizon=3,
        action_dim=1,
        max_divergence_norm=1.0,
    )
    cache = WarmStartCache(config)
    cache.update(np.array([[0.0], [1.0], [2.0]]))

    # Small divergence: ok
    assert cache.verify_consistency(np.array([0.2])) is True
    assert cache.is_active is True

    # Large divergence: invalidates
    assert cache.verify_consistency(np.array([2.5])) is False
    assert cache.is_active is False
    assert cache.get_warm_start() is None
