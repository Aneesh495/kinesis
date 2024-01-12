"""Declared nominal and frozen physical shift profiles, physics mutators, and probe verifiers."""

from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np

from kinesis.contracts import ActionSpec, ObservationSpec, Transition
from kinesis.contracts.types import ShiftCategory
from kinesis.envs.base import Environment
from kinesis.envs.dm_control_adapter import DMControlAdapter
from kinesis.observations.wrapper import PartialObservationWrapper


@dataclass(frozen=True)
class ShiftProfile:
    """Immutable specification of a physical or observational environment shift."""
    category: ShiftCategory
    name: str
    description: str
    actuator_gear_scale: float = 1.0
    geom_friction_scale: float = 1.0
    body_mass_scale: float = 1.0
    dof_damping_scale: float = 1.0
    actuator_delay: int = 0
    observation_delay: int = 0
    observation_noise_std: float = 0.0
    observation_dropout_prob: float = 0.0


# Declared frozen profiles
NOMINAL_PROFILE = ShiftProfile(
    category=ShiftCategory.NOMINAL,
    name="nominal",
    description="Unperturbed nominal DeepMind Control Suite dynamics and observation channel",
)

ACTUATOR_ATTENUATION_PROFILE = ShiftProfile(
    category=ShiftCategory.ACTUATOR_DEADBAND,
    name="actuator_attenuation",
    description="Actuator torque capacity reduced by 50 percent",
    actuator_gear_scale=0.5,
)

ACTUATOR_DELAY_PROFILE = ShiftProfile(
    category=ShiftCategory.ACTUATOR_DELAY,
    name="actuator_delay",
    description="Two-step transport latency in actuator torque application",
    actuator_delay=2,
)

FRICTION_PERTURBATION_PROFILE = ShiftProfile(
    category=ShiftCategory.FRICTION_PERTURBATION,
    name="friction_reduction",
    description="Contact and joint sliding friction reduced to 25 percent (slip regime)",
    geom_friction_scale=0.25,
    dof_damping_scale=0.2,
)

MASS_PERTURBATION_PROFILE = ShiftProfile(
    category=ShiftCategory.MASS_INERTIA,
    name="mass_increase",
    description="Rigid body masses augmented by 80 percent (heavy payload regime)",
    body_mass_scale=1.8,
)

SENSOR_DEGRADATION_PROFILE = ShiftProfile(
    category=ShiftCategory.SENSOR_DROPOUT,
    name="sensor_degradation",
    description="Sensory delay of 2 steps, 15 percent channel dropout, and Gaussian noise std 0.08",
    observation_delay=2,
    observation_noise_std=0.08,
    observation_dropout_prob=0.15,
)

COMBINED_STRESS_PROFILE = ShiftProfile(
    category=ShiftCategory.COMBINED_STRESS,
    name="combined_stress",
    description="Combined multi-stress: 40 percent actuator loss, 1-step delay, reduced friction, mass increase, and noisy sensors",
    actuator_gear_scale=0.6,
    actuator_delay=1,
    geom_friction_scale=0.35,
    body_mass_scale=1.4,
    observation_delay=1,
    observation_noise_std=0.05,
    observation_dropout_prob=0.10,
)

ALL_SHIFT_PROFILES: dict[str, ShiftProfile] = {
    "nominal": NOMINAL_PROFILE,
    "actuator_attenuation": ACTUATOR_ATTENUATION_PROFILE,
    "actuator_delay": ACTUATOR_DELAY_PROFILE,
    "friction_reduction": FRICTION_PERTURBATION_PROFILE,
    "mass_increase": MASS_PERTURBATION_PROFILE,
    "sensor_degradation": SENSOR_DEGRADATION_PROFILE,
    "combined_stress": COMBINED_STRESS_PROFILE,
}


class ActuatorDelayWrapper(Environment):
    """Applies transport delay to executed actions."""

    def __init__(self, env: Environment, delay_steps: int = 1) -> None:
        self._env = env
        self._delay_steps = max(0, delay_steps)
        self._action_buffer: deque[np.ndarray] = deque(maxlen=self._delay_steps + 1)

    @property
    def task_name(self) -> str:
        return self._env.task_name

    @property
    def action_spec(self) -> ActionSpec:
        return self._env.action_spec

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._env.observation_spec

    def reset(self, seed: int | None = None) -> Transition:
        self._action_buffer.clear()
        zero_act = self._env.action_spec.zero_action()
        for _ in range(self._delay_steps + 1):
            self._action_buffer.append(zero_act)
        return self._env.reset(seed=seed)

    def step(self, action: np.ndarray) -> Transition:
        self._action_buffer.append(action.copy())
        # The oldest action in buffer takes effect
        delayed_act = self._action_buffer[0]
        return self._env.step(delayed_act)

    def get_physics_probe(self) -> dict[str, float]:
        return self._env.get_physics_probe()

    def close(self) -> None:
        self._env.close()


def build_shifted_environment(
    task_name: str,
    profile_name: str = "nominal",
    action_repeat: int = 2,
    seed: int = 42,
) -> Environment:
    """Construct environment and apply declared physical and observational perturbations."""
    if profile_name not in ALL_SHIFT_PROFILES:
        raise ValueError(
            f"Unknown shift profile '{profile_name}'. Legal profiles: {list(ALL_SHIFT_PROFILES.keys())}"
        )

    profile = ALL_SHIFT_PROFILES[profile_name]

    def _apply_physics_mutations(physics: Any) -> None:
        model = physics.model
        if profile.actuator_gear_scale != 1.0 and model.actuator_gear.size > 0:
            model.actuator_gear[:] *= profile.actuator_gear_scale
        if profile.geom_friction_scale != 1.0 and model.geom_friction.size > 0:
            model.geom_friction[:, 0] *= profile.geom_friction_scale
        if profile.body_mass_scale != 1.0 and model.body_mass.size > 0:
            model.body_mass[:] *= profile.body_mass_scale
        if profile.dof_damping_scale != 1.0 and model.dof_damping.size > 0:
            model.dof_damping[:] *= profile.dof_damping_scale

    raw_env: Environment = DMControlAdapter(
        task_name=task_name,
        action_repeat=action_repeat,
        seed=seed,
        physics_mutator=_apply_physics_mutations,
    )

    if profile.actuator_delay > 0:
        raw_env = ActuatorDelayWrapper(raw_env, delay_steps=profile.actuator_delay)

    wrapped_env: Environment = PartialObservationWrapper(
        env=raw_env,
        observation_delay=profile.observation_delay,
        noise_std=profile.observation_noise_std,
        dropout_prob=profile.observation_dropout_prob,
        seed=seed,
    )

    return wrapped_env


def probe_dynamics_divergence(
    task_name: str,
    shift_profile_name: str,
    num_steps: int = 25,
    seed: int = 42,
) -> dict[str, float]:
    """Execute identical action sequences on nominal vs shifted environments to measure divergence."""
    env_nom = build_shifted_environment(task_name, "nominal", seed=seed)
    env_shift = build_shifted_environment(task_name, shift_profile_name, seed=seed)

    t_nom = env_nom.reset(seed=seed)
    t_shift = env_shift.reset(seed=seed)

    rng = np.random.default_rng(seed)
    obs_spec = env_nom.observation_spec
    act_spec = env_nom.action_spec

    total_obs_divergence = 0.0
    total_reward_diff = 0.0

    for _ in range(num_steps):
        act = act_spec.sample_uniform(rng)
        t_nom = env_nom.step(act)
        t_shift = env_shift.step(act)

        flat_nom = obs_spec.flatten_public(t_nom.observation)
        flat_shift = obs_spec.flatten_public(t_shift.observation)

        diff = float(np.linalg.norm(flat_nom - flat_shift))
        total_obs_divergence += diff
        total_reward_diff += abs(t_nom.reward - t_shift.reward)

    env_nom.close()
    env_shift.close()

    return {
        "mean_obs_divergence": total_obs_divergence / num_steps,
        "mean_reward_difference": total_reward_diff / num_steps,
    }
