"""Partial observation wrapper enforcing sensory delays, dropout, and measurement noise."""

from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np

from kinesis.contracts import ObservationChannel, ObservationSpec, Transition
from kinesis.envs.base import Environment


@dataclass
class RunningNormalizationStats:
    """Welford's algorithm for online streaming mean and variance tracking."""
    count: int = 0
    mean: np.ndarray | None = None
    M2: np.ndarray | None = None

    def update(self, x: np.ndarray) -> None:
        x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
        if self.mean is None or self.M2 is None:
            self.mean = np.zeros_like(x_flat)
            self.M2 = np.zeros_like(x_flat)
            self.count = 0

        self.count += 1
        delta = x_flat - self.mean
        self.mean += delta / self.count
        delta2 = x_flat - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2 or self.M2 is None:
            return np.ones_like(self.mean) if self.mean is not None else np.array([1.0])
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 1e-6))

    def normalize(self, x: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        if self.mean is None or self.count < 2:
            return x
        normalized = (x - self.mean) / self.std
        return np.clip(normalized, -clip_range, clip_range).astype(np.float32)


class PartialObservationWrapper(Environment):
    """Enforces rigorous partial observability: delays, noise, dropout, and history."""

    def __init__(
        self,
        env: Environment,
        observation_delay: int = 0,
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        self._env = env
        self._delay = max(0, observation_delay)
        self._noise_std = max(0.0, noise_std)
        self._dropout_prob = min(max(0.0, dropout_prob), 1.0)
        self._rng = np.random.default_rng(seed)

        self._delay_buffer: deque[dict[str, np.ndarray]] = deque(maxlen=self._delay + 1)
        self._normalization_stats = RunningNormalizationStats()

    @property
    def task_name(self) -> str:
        return self._env.task_name

    @property
    def action_spec(self) -> Any:
        return self._env.action_spec

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._env.observation_spec

    def _perturb_observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply noise and sensor dropout strictly to public observation channels."""
        perturbed = {}
        for k, v in obs.items():
            arr = v.copy()
            # Additive zero-mean Gaussian noise
            if self._noise_std > 0.0:
                noise = self._rng.normal(0.0, self._noise_std, size=arr.shape).astype(np.float32)
                arr = arr + noise

            # Sensor channel dropout
            if self._dropout_prob > 0.0 and self._rng.uniform(0.0, 1.0) < self._dropout_prob:
                arr = np.zeros_like(arr)

            perturbed[k] = arr
        return perturbed

    def reset(self, seed: int | None = None) -> Transition:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        raw_transition = self._env.reset(seed=seed)

        # Clear and pre-fill delay buffer
        self._delay_buffer.clear()
        initial_obs = self._perturb_observation(raw_transition.observation)
        for _ in range(self._delay + 1):
            self._delay_buffer.append(initial_obs)

        # The delayed observation is the oldest element in the buffer
        delayed_obs = self._delay_buffer[0]

        # Update running normalization
        flat_obs = self._env.observation_spec.flatten_public(delayed_obs)
        self._normalization_stats.update(flat_obs)

        return Transition(
            transition_id=raw_transition.transition_id,
            episode_id=raw_transition.episode_id,
            step_index=raw_transition.step_index,
            step_type=raw_transition.step_type,
            observation=delayed_obs,
            action=raw_transition.action,
            reward=raw_transition.reward,
            constraint_cost=raw_transition.constraint_cost,
            discount=raw_transition.discount,
            time_delta=raw_transition.time_delta,
            simulator_substeps=raw_transition.simulator_substeps,
            privileged_diagnostics=raw_transition.privileged_diagnostics,
        )

    def step(self, action: np.ndarray) -> Transition:
        raw_transition = self._env.step(action)
        curr_obs = self._perturb_observation(raw_transition.observation)
        self._delay_buffer.append(curr_obs)

        delayed_obs = self._delay_buffer[0]

        flat_obs = self._env.observation_spec.flatten_public(delayed_obs)
        self._normalization_stats.update(flat_obs)

        return Transition(
            transition_id=raw_transition.transition_id,
            episode_id=raw_transition.episode_id,
            step_index=raw_transition.step_index,
            step_type=raw_transition.step_type,
            observation=delayed_obs,
            action=raw_transition.action,
            reward=raw_transition.reward,
            constraint_cost=raw_transition.constraint_cost,
            discount=raw_transition.discount,
            time_delta=raw_transition.time_delta,
            simulator_substeps=raw_transition.simulator_substeps,
            privileged_diagnostics=raw_transition.privileged_diagnostics,
        )

    def get_physics_probe(self) -> dict[str, float]:
        return self._env.get_physics_probe()

    def close(self) -> None:
        self._env.close()
