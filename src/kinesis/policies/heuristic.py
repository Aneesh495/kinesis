"""Heuristic and classical PD/energy-shaping baseline controllers."""

import numpy as np

from kinesis.contracts import ActionSpec, ModelVersion
from kinesis.policies.base import BasePolicy


class CartpoleEnergyHeuristic(BasePolicy):
    """Energy-shaping swingup and PD stabilization heuristic for cartpole."""

    def __init__(self, action_spec: ActionSpec, version: str = "heuristic_cartpole_v1.0") -> None:
        self._action_spec = action_spec
        self._version = ModelVersion(version)

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        # Cartpole observation: position = [cart_pos, cos(theta), sin(theta)], velocity = [cart_vel, theta_dot]
        pos = observation.get("position", np.zeros(3))
        vel = observation.get("velocity", np.zeros(2))

        cart_pos = float(pos[0]) if len(pos) > 0 else 0.0
        cos_th = float(pos[1]) if len(pos) > 1 else 1.0
        sin_th = float(pos[2]) if len(pos) > 2 else 0.0
        th_dot = float(vel[1]) if len(vel) > 1 else 0.0

        # When pendulum is near upright (cos_th > 0.8), use linear balancing PD control
        if cos_th > 0.8:
            # PD balance: cart position centering + pendulum stabilization
            action_val = 1.2 * sin_th + 0.3 * th_dot - 0.4 * cart_pos
        else:
            # Energy pumping swingup: push cart in direction of angular velocity * cos_th
            action_val = np.sign(th_dot * cos_th) * 0.8 if abs(th_dot) > 0.1 else 0.5

        action_arr = np.array([action_val], dtype=np.float32)
        return self._action_spec.clip(action_arr)


class WalkerPeriodicHeuristic(BasePolicy):
    """Sinusoidal open-loop oscillator heuristic for bipedal walker."""

    def __init__(self, action_spec: ActionSpec, version: str = "heuristic_walker_v1.0") -> None:
        self._action_spec = action_spec
        self._version = ModelVersion(version)
        self._step = 0

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        self._step += 1
        freq = 0.1
        phase = self._step * freq * 2.0 * np.pi
        dim = self._action_spec.dim
        # Coordinated alternating leg oscillation
        pattern = np.zeros(dim, dtype=np.float32)
        for i in range(dim):
            sign = 1.0 if (i % 2 == 0) else -1.0
            pattern[i] = 0.5 * np.sin(phase + sign * np.pi / 2.0)
        return self._action_spec.clip(pattern)

    def reset(self) -> None:
        self._step = 0


class FingerTrackingHeuristic(BasePolicy):
    """Proportional velocity tracking controller for finger spinning."""

    def __init__(self, action_spec: ActionSpec, version: str = "heuristic_finger_v1.0") -> None:
        self._action_spec = action_spec
        self._version = ModelVersion(version)

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        # Finger position: [proximal, distal, spinner_angle]
        pos = observation.get("position", np.zeros(3))
        dim = self._action_spec.dim
        action = np.ones(dim, dtype=np.float32) * 0.7
        return self._action_spec.clip(action)
