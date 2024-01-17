"""Closed-loop latent planner policy with deadline monitoring and safe fallback."""

import time
from typing import Any
import numpy as np
import torch

from kinesis.contracts import ActionSpec, ModelVersion, ObservationSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel
from kinesis.planning.cem import CrossEntropyMethodPlanner, PlanningResult
from kinesis.planning.mppi import ModelPredictivePathIntegralPlanner
from kinesis.policies.base import BasePolicy
from kinesis.policies.fallback import SafeFallbackPolicy
from kinesis.uncertainty.ensemble import WorldModelEnsemble


class LatentPlannerPolicy(BasePolicy):
    """Closed-loop policy executing online latent trajectory optimization under partial observability."""

    def __init__(
        self,
        planner: CrossEntropyMethodPlanner | ModelPredictivePathIntegralPlanner,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        timeout_sec: float = 0.5,
        version: str = "latent_planner_v1.0",
    ) -> None:
        self.planner = planner
        self.observation_spec = observation_spec
        self._action_spec = action_spec
        self.timeout_sec = timeout_sec
        self._version = ModelVersion(version)
        self.fallback_policy = SafeFallbackPolicy(action_spec)

        self._belief_states: list[RSSMState] | None = None
        self._last_action: np.ndarray | None = None
        self.last_planning_result: PlanningResult | None = None

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    def reset(self) -> None:
        self._belief_states = None
        self._last_action = None
        self.planner.reset_warm_start()

    def update_beliefs(self, observation: dict[str, np.ndarray]) -> list[RSSMState]:
        """Update belief state across all world models in the planner."""
        flat_obs = self.observation_spec.flatten_public(observation)
        obs_t = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)

        models = (
            self.planner.ensemble.models
            if isinstance(self.planner.ensemble, WorldModelEnsemble)
            else [self.planner.ensemble]
        )

        device = obs_t.device
        act_np = self._last_action if self._last_action is not None else self._action_spec.zero_action()
        act_t = torch.tensor(act_np, dtype=torch.float32, device=device).unsqueeze(0)

        updated: list[RSSMState] = []
        for i, m in enumerate(models):
            embed = m.encoder(obs_t)
            if self._belief_states is None:
                init_s = m.rssm.initial_state(1, device)
                zero_act = torch.zeros(1, self._action_spec.dim, device=device)
                _, post_s = m.rssm.observe_step(init_s, zero_act, embed)
            else:
                prev_s = self._belief_states[i]
                _, post_s = m.rssm.observe_step(prev_s, act_t, embed)
            updated.append(post_s)

        self._belief_states = updated
        return updated

    def act(self, observation: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        t_start = time.perf_counter()
        try:
            beliefs = self.update_beliefs(observation)

            # Check if planning time exceeded deadline
            plan_res = self.planner.plan(beliefs)
            self.last_planning_result = plan_res

            elapsed = time.perf_counter() - t_start
            if elapsed > self.timeout_sec:
                self.fallback_policy.record_trigger("timeout")
                selected_action = self.fallback_policy.act(observation)
            else:
                selected_action = plan_res.selected_action

            if not np.all(np.isfinite(selected_action)):
                self.fallback_policy.record_trigger("nonfinite")
                selected_action = self.fallback_policy.act(observation)

        except Exception as e:
            self.fallback_policy.record_trigger("model_failure")
            selected_action = self.fallback_policy.act(observation)

        self._last_action = selected_action
        return self._action_spec.clip(selected_action)
