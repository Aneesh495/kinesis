"""DeepMind Control Suite environment adapter with substep accounting and physics probing."""

from collections.abc import Callable
import hashlib
from typing import Any
import numpy as np
from dm_control import suite

from kinesis.contracts import (
    ActionSpec,
    EpisodeID,
    Modality,
    ModelVersion,
    ObservationChannel,
    ObservationSpec,
    StepType,
    Transition,
    TransitionID,
)
from kinesis.envs.base import Environment


class DMControlAdapter(Environment):
    """Adapter wrapping dm_control physics simulations into Kinesis contracts."""

    SUPPORTED_TASKS: dict[str, tuple[str, str]] = {
        "cartpole_swingup": ("cartpole", "swingup"),
        "walker_walk": ("walker", "walk"),
        "finger_spin": ("finger", "spin"),
    }

    def __init__(
        self,
        task_name: str,
        action_repeat: int = 2,
        seed: int = 42,
        physics_mutator: Callable[[Any], None] | None = None,
    ) -> None:
        if task_name not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task_name}'. Supported tasks: {list(self.SUPPORTED_TASKS.keys())}"
            )

        self._task_name = task_name
        self._action_repeat = max(1, action_repeat)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._physics_mutator = physics_mutator

        domain, task = self.SUPPORTED_TASKS[task_name]
        self._dm_env = suite.load(domain, task, task_kwargs={"random": seed})
        if self._physics_mutator is not None:
            self._physics_mutator(self._dm_env.physics)

        # Build ActionSpec
        dm_act_spec = self._dm_env.action_spec()
        self._action_spec = ActionSpec(
            shape=dm_act_spec.shape,
            minimum=np.array(dm_act_spec.minimum, dtype=np.float32),
            maximum=np.array(dm_act_spec.maximum, dtype=np.float32),
            action_repeat=self._action_repeat,
            units="normalized_torque",
        )

        # Build ObservationSpec
        sample_step = self._dm_env.reset()
        channels = []
        for name, val in sample_step.observation.items():
            arr = np.asarray(val, dtype=np.float32)
            channels.append(
                ObservationChannel(
                    name=name,
                    shape=arr.shape,
                    modality=Modality.PROPRIOCEPTION,
                    units="dm_control_units",
                )
            )
        self._observation_spec = ObservationSpec(channels=tuple(channels))

        self._current_episode_id = EpisodeID(f"ep_{self._task_name}_{self._seed}_0")
        self._step_count = 0
        self._episode_count = 0
        self._sim_timestep = float(self._dm_env.physics.model.opt.timestep)

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._observation_spec

    @property
    def physics(self) -> Any:
        return self._dm_env.physics

    def get_model_hash(self) -> str:
        """Deterministic hash of MuJoCo model structure and parameters."""
        model = self._dm_env.physics.model
        hasher = hashlib.sha256()
        hasher.update(self._task_name.encode("utf-8"))
        for field_name in ["body_mass", "geom_friction", "actuator_gear", "dof_damping"]:
            arr = getattr(model, field_name, None)
            if arr is not None:
                hasher.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return hasher.hexdigest()

    def reset(self, seed: int | None = None) -> Transition:
        """Reset environment physics and return initial Transition."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        self._episode_count += 1
        self._current_episode_id = EpisodeID(f"ep_{self._task_name}_{self._seed}_{self._episode_count}")
        self._step_count = 0

        dm_step = self._dm_env.reset()
        obs = {k: np.asarray(v, dtype=np.float32) for k, v in dm_step.observation.items()}
        zero_action = self._action_spec.zero_action()

        return Transition(
            transition_id=TransitionID(f"{self._current_episode_id}_s0"),
            episode_id=self._current_episode_id,
            step_index=0,
            step_type=StepType.FIRST,
            observation=obs,
            action=zero_action,
            reward=0.0,
            constraint_cost=0.0,
            discount=1.0,
            time_delta=float(self._sim_timestep * self._action_repeat),
            simulator_substeps=self._action_repeat,
            privileged_diagnostics=self.get_physics_probe_array(),
        )

    def step(self, action: np.ndarray) -> Transition:
        """Step environment dynamics with repeated actions and substep accounting."""
        clipped_action = self._action_spec.clip(action)
        self._step_count += 1

        total_reward = 0.0
        step_type = StepType.MID
        dm_step = None

        # Execute action repeat substeps
        for _ in range(self._action_repeat):
            dm_step = self._dm_env.step(clipped_action)
            if dm_step.reward is not None:
                total_reward += float(dm_step.reward)
            if dm_step.last():
                step_type = StepType.LAST
                break

        assert dm_step is not None
        discount = float(dm_step.discount) if dm_step.discount is not None else 1.0
        if step_type == StepType.LAST:
            discount = 0.0

        obs = {k: np.asarray(v, dtype=np.float32) for k, v in dm_step.observation.items()}

        # Simple control effort constraint cost (norm of action)
        constraint_cost = float(np.sum(np.square(clipped_action)) * 0.01)

        return Transition(
            transition_id=TransitionID(f"{self._current_episode_id}_s{self._step_count}"),
            episode_id=self._current_episode_id,
            step_index=self._step_count,
            step_type=step_type,
            observation=obs,
            action=clipped_action,
            reward=total_reward,
            constraint_cost=constraint_cost,
            discount=discount,
            time_delta=float(self._sim_timestep * self._action_repeat),
            simulator_substeps=self._action_repeat,
            privileged_diagnostics=self.get_physics_probe_array(),
        )

    def get_physics_probe(self) -> dict[str, float]:
        """Extract physical parameters (mass, friction, damping, gear)."""
        model = self._dm_env.physics.model
        total_mass = float(np.sum(model.body_mass))
        mean_geom_friction = float(np.mean(model.geom_friction[:, 0]))
        mean_dof_damping = float(np.mean(model.dof_damping)) if model.dof_damping.size > 0 else 0.0
        mean_actuator_gear = float(np.mean(model.actuator_gear[:, 0])) if model.actuator_gear.size > 0 else 1.0

        return {
            "total_mass": total_mass,
            "mean_geom_friction": mean_geom_friction,
            "mean_dof_damping": mean_dof_damping,
            "mean_actuator_gear": mean_actuator_gear,
        }

    def get_physics_probe_array(self) -> dict[str, np.ndarray]:
        probe = self.get_physics_probe()
        return {k: np.array([v], dtype=np.float32) for k, v in probe.items()}

    def close(self) -> None:
        pass
