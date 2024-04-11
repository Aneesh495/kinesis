"""Deduplicated latent trajectory rollout hash cache with LRU eviction."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass
class CachedRollout:
    """Cached output from an imagined latent trajectory rollout."""
    rewards: np.ndarray
    total_reward: float
    uncertainties: np.ndarray | None
    terminal_state: np.ndarray | None
    access_count: int = 1


@dataclass
class CacheStatistics:
    """Telemetry and operational metrics for the rollout cache."""
    hits: int
    misses: int
    evictions: int
    current_size: int
    max_capacity: int
    hit_rate: float


class LatentRolloutCache:
    """Quantized hash cache for imagined world model trajectory rollouts."""

    def __init__(self, max_capacity: int = 5000, quantization_decimals: int = 3) -> None:
        self.max_capacity = int(max_capacity)
        self.quantization_decimals = int(quantization_decimals)
        self._cache: OrderedDict[str, CachedRollout] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _hash_key(self, initial_state: np.ndarray, actions: np.ndarray) -> str:
        """Construct deterministic hash key from quantized state and action trajectory."""
        s_quant = np.round(np.asarray(initial_state, dtype=np.float64), self.quantization_decimals)
        a_quant = np.round(np.asarray(actions, dtype=np.float64), self.quantization_decimals)

        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(s_quant.tobytes())
        hasher.update(a_quant.tobytes())
        return hasher.hexdigest()

    def get(self, initial_state: np.ndarray, actions: np.ndarray) -> CachedRollout | None:
        """Retrieve cached rollout if matching quantized trajectory exists."""
        key = self._hash_key(initial_state, actions)
        if key in self._cache:
            self.hits += 1
            entry = self._cache[key]
            entry.access_count += 1
            self._cache.move_to_end(key)
            return entry

        self.misses += 1
        return None

    def put(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        uncertainties: np.ndarray | None = None,
        terminal_state: np.ndarray | None = None,
    ) -> None:
        """Insert newly simulated trajectory into LRU cache."""
        key = self._hash_key(initial_state, actions)
        r_arr = np.asarray(rewards, dtype=np.float64)
        u_arr = np.asarray(uncertainties, dtype=np.float64) if uncertainties is not None else None
        s_term = np.asarray(terminal_state, dtype=np.float64) if terminal_state is not None else None

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key].rewards = r_arr
            self._cache[key].total_reward = float(np.sum(r_arr))
            self._cache[key].uncertainties = u_arr
            self._cache[key].terminal_state = s_term
            return

        if len(self._cache) >= self.max_capacity:
            # Evict oldest LRU item
            self._cache.popitem(last=False)
            self.evictions += 1

        self._cache[key] = CachedRollout(
            rewards=r_arr,
            total_reward=float(np.sum(r_arr)),
            uncertainties=u_arr,
            terminal_state=s_term,
            access_count=1,
        )

    def stats(self) -> CacheStatistics:
        """Compute hit rate and telemetry."""
        total_requests = self.hits + self.misses
        hit_ratio = float(self.hits / total_requests) if total_requests > 0 else 0.0
        return CacheStatistics(
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            current_size=len(self._cache),
            max_capacity=self.max_capacity,
            hit_rate=hit_ratio,
        )

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
