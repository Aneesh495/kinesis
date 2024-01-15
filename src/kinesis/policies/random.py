"""Uniform random baseline policy."""

import numpy as np

from kinesis.contracts import ActionSpec, ModelVersion
from kinesis.policies.base import BasePolicy


class RandomPolicy(BasePolicy):
    """Uniform random action sampling within declared ActionSpec bounds."""

    def __init__(self, action_spec: ActionSpec, seed: int = 42, version: str = "random_v1.0") -> None:
        self._action_spec = action_spec
        self._rng = np.random.default_rng(seed)
        self._version = ModelVersion(version)

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return self._action_spec.zero_action()
        return self._action_spec.sample_uniform(self._rng)

    def reset(self) -> None:
        pass
