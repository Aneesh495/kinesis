"""Bounded immutable replay buffer with episode-safe sequence sampling and capacity eviction."""

from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from kinesis.contracts import Episode, Transition
from kinesis.replay.chunk import ReplayChunk


@dataclass(frozen=True)
class ReplayBufferStats:
    """Runtime capacity and usage statistics for replay buffer."""
    total_transitions: int
    total_episodes: int
    capacity: int
    evicted_chunks_count: int
    num_chunks: int


class BoundedReplayBuffer:
    """Bounded replay buffer storing immutable chunks with episode-safe sequence sampling."""

    def __init__(self, capacity_transitions: int = 100_000, chunk_size: int = 100) -> None:
        if capacity_transitions < 1:
            raise ValueError(f"capacity_transitions must be >= 1, got {capacity_transitions}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        self._capacity = capacity_transitions
        self._chunk_size = chunk_size

        self._chunks: deque[ReplayChunk] = deque()
        self._current_chunk_transitions: list[Transition] = []
        self._chunk_counter = 0
        self._evicted_chunks_count = 0

        # Episode store: episode_id -> list of transitions
        self._episodes: dict[str, list[Transition]] = {}
        # List of completed or ongoing episode IDs that have at least one transition
        self._episode_id_order: list[str] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_transitions(self) -> int:
        return sum(c.num_transitions for c in self._chunks) + len(self._current_chunk_transitions)

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    def add_transition(self, transition: Transition) -> None:
        """Append an individual transition to the active chunk and episode index."""
        ep_id = str(transition.episode_id)
        if ep_id not in self._episodes:
            self._episodes[ep_id] = []
            self._episode_id_order.append(ep_id)

        self._episodes[ep_id].append(transition)
        self._current_chunk_transitions.append(transition)

        # Seal chunk if full
        if len(self._current_chunk_transitions) >= self._chunk_size:
            self._seal_active_chunk()

        # Enforce capacity
        self._enforce_capacity()

    def add_episode(self, episode: Episode) -> None:
        """Add an entire complete episode directly."""
        for t in episode.transitions:
            self.add_transition(t)

    def _seal_active_chunk(self) -> None:
        if not self._current_chunk_transitions:
            return
        self._chunk_counter += 1
        chunk_id = f"chunk_{self._chunk_counter:06d}"
        chunk = ReplayChunk(chunk_id=chunk_id, transitions=self._current_chunk_transitions)
        self._chunks.append(chunk)
        self._current_chunk_transitions = []

    def _enforce_capacity(self) -> None:
        """Evict oldest immutable chunks if total transitions exceed capacity."""
        while self.total_transitions > self._capacity and len(self._chunks) > 0:
            evicted = self._chunks.popleft()
            self._evicted_chunks_count += 1
            # Clean up episodes fully evicted from all active chunks
            active_ep_ids = set()
            for c in self._chunks:
                active_ep_ids.update(c.episode_ids)
            for t in self._current_chunk_transitions:
                active_ep_ids.add(str(t.episode_id))

            # Evict dead episodes from episode store
            dead_episodes = [ep_id for ep_id in self._episodes if ep_id not in active_ep_ids]
            for ep_id in dead_episodes:
                del self._episodes[ep_id]
                if ep_id in self._episode_id_order:
                    self._episode_id_order.remove(ep_id)

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Sample batch_size sequences of length sequence_length without crossing episode resets."""
        if rng is None:
            rng = np.random.default_rng()

        # Filter eligible episodes with length >= sequence_length
        eligible_episodes = [
            ep_id for ep_id, transitions in self._episodes.items()
            if len(transitions) >= sequence_length
        ]

        if not eligible_episodes:
            raise ValueError(
                f"No eligible episodes with length >= {sequence_length} in replay buffer (total transitions={self.total_transitions})"
            )

        batch_obs_keys: set[str] = set(self._episodes[eligible_episodes[0]][0].observation.keys())
        batch_obs: dict[str, list[list[np.ndarray]]] = {k: [] for k in batch_obs_keys}
        batch_actions: list[list[np.ndarray]] = []
        batch_rewards: list[list[float]] = []
        batch_discounts: list[list[float]] = []
        batch_costs: list[list[float]] = []
        batch_is_first: list[list[float]] = []

        for _ in range(batch_size):
            selected_ep_id = rng.choice(eligible_episodes)
            transitions = self._episodes[selected_ep_id]
            max_start = len(transitions) - sequence_length
            start_idx = int(rng.integers(0, max_start + 1))

            seq = transitions[start_idx : start_idx + sequence_length]

            # Collect sequence elements
            seq_acts = []
            seq_rews = []
            seq_discs = []
            seq_costs = []
            seq_first = []
            per_key_obs: dict[str, list[np.ndarray]] = {k: [] for k in batch_obs_keys}

            for step_in_seq, t in enumerate(seq):
                for k in batch_obs_keys:
                    per_key_obs[k].append(t.observation[k])
                seq_acts.append(t.action)
                seq_rews.append(t.reward)
                seq_discs.append(t.discount)
                seq_costs.append(t.constraint_cost)
                # First step of sequence is marked as first if it is the first step of episode
                seq_first.append(1.0 if t.step_type.is_first() else 0.0)

            for k in batch_obs_keys:
                batch_obs[k].append(per_key_obs[k])
            batch_actions.append(seq_acts)
            batch_rewards.append(seq_rews)
            batch_discounts.append(seq_discs)
            batch_costs.append(seq_costs)
            batch_is_first.append(seq_first)

        # Convert to PyTorch tensors: [B, L, ...]
        tensor_obs = {
            k: torch.tensor(np.array(batch_obs[k]), dtype=torch.float32)
            for k in batch_obs_keys
        }
        tensor_actions = torch.tensor(np.array(batch_actions), dtype=torch.float32)
        tensor_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)
        tensor_discounts = torch.tensor(np.array(batch_discounts), dtype=torch.float32)
        tensor_costs = torch.tensor(np.array(batch_costs), dtype=torch.float32)
        tensor_is_first = torch.tensor(np.array(batch_is_first), dtype=torch.float32)

        return {
            "observations": tensor_obs,
            "actions": tensor_actions,
            "rewards": tensor_rewards,
            "discounts": tensor_discounts,
            "constraint_costs": tensor_costs,
            "is_first": tensor_is_first,
        }

    def get_stats(self) -> ReplayBufferStats:
        return ReplayBufferStats(
            total_transitions=self.total_transitions,
            total_episodes=self.total_episodes,
            capacity=self._capacity,
            evicted_chunks_count=self._evicted_chunks_count,
            num_chunks=len(self._chunks),
        )
