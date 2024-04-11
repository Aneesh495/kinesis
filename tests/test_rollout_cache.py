"""Unit tests for deduplicated latent trajectory rollout cache."""

import numpy as np
import pytest

from kinesis.planning.rollout_cache import LatentRolloutCache, CachedRollout, CacheStatistics


def test_rollout_cache_put_get_hit():
    """Verify storing and retrieving identical trajectory hits the cache."""
    cache = LatentRolloutCache(max_capacity=100, quantization_decimals=3)

    s0 = np.array([0.1234, -0.5678])
    actions = np.array([[0.1, 0.2], [-0.1, 0.4]])
    rewards = np.array([1.0, 1.5])
    unc = np.array([0.05, 0.08])

    # Initial miss
    assert cache.get(s0, actions) is None

    # Put
    cache.put(s0, actions, rewards, unc)

    # Hit
    hit = cache.get(s0, actions)
    assert hit is not None
    assert isinstance(hit, CachedRollout)
    assert hit.total_reward == 2.5
    np.testing.assert_allclose(hit.rewards, rewards)
    np.testing.assert_allclose(hit.uncertainties, unc)

    # Approximate match within quantization threshold
    s0_approx = np.array([0.1231, -0.5681])
    hit_approx = cache.get(s0_approx, actions)
    assert hit_approx is not None

    stats = cache.stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.hit_rate > 0.6


def test_rollout_cache_lru_eviction():
    """Verify oldest entries are evicted when exceeding maximum capacity."""
    cache = LatentRolloutCache(max_capacity=2, quantization_decimals=2)

    s0 = np.array([0.0])
    a1 = np.array([[1.0]])
    a2 = np.array([[2.0]])
    a3 = np.array([[3.0]])

    cache.put(s0, a1, np.array([1.0]))
    cache.put(s0, a2, np.array([2.0]))
    assert cache.stats().current_size == 2

    # Put third entry: triggers eviction of a1
    cache.put(s0, a3, np.array([3.0]))
    stats = cache.stats()
    assert stats.current_size == 2
    assert stats.evictions == 1

    # a1 should be evicted (miss)
    assert cache.get(s0, a1) is None
    # a2 and a3 should be present
    assert cache.get(s0, a2) is not None
    assert cache.get(s0, a3) is not None
