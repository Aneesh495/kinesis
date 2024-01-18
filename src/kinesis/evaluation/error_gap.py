"""Analysis of the gap between one-step model accuracy and multi-step closed-loop decisions."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from kinesis.contracts import Episode
from kinesis.models.world_model import WorldModel


@dataclass(frozen=True)
class ErrorGapAnalysisResult:
    """Quantitative breakdown of prediction vs closed-loop decision errors."""
    task_name: str
    shift_name: str
    one_step_recon_mse: float
    one_step_reward_mse: float
    horizon_errors: dict[int, float]  # horizon -> mean rollout MSE
    mean_predicted_return: float
    mean_actual_return: float
    decision_regret: float  # predicted_return - actual_return


class PredictionControlErrorGapAnalyzer:
    """Audits divergence between one-step model fidelity and closed-loop control outcomes."""

    def __init__(self, world_model: WorldModel) -> None:
        self.world_model = world_model

    def analyze_episode(
        self,
        episode: Episode,
        shift_name: str = "nominal",
        max_horizon: int = 12,
    ) -> ErrorGapAnalysisResult:
        """Evaluate open-loop prediction error vs closed-loop return across episode transitions."""
        transitions = episode.transitions
        T = len(transitions)
        if T < 2:
            raise ValueError("Episode too short for error gap analysis")

        # Collect observation vectors and actions
        obs_spec = episode.transitions[0].observation
        flat_dim = sum(v.size for v in obs_spec.values())

        obs_arrays = []
        act_arrays = []
        rews = []
        for t in transitions:
            flat = np.concatenate([v.reshape(-1) for v in t.observation.values()]).astype(np.float32)
            obs_arrays.append(flat)
            act_arrays.append(t.action)
            rews.append(t.reward)

        obs_t = torch.tensor(np.array(obs_arrays), dtype=torch.float32).unsqueeze(0)  # [1, T, obs_dim]
        act_t = torch.tensor(np.array(act_arrays), dtype=torch.float32).unsqueeze(0)  # [1, T, act_dim]
        is_first = torch.zeros(1, T)
        is_first[:, 0] = 1.0

        with torch.no_grad():
            embeds = self.world_model.encoder(obs_t)
            prior_states, post_states = self.world_model.rssm.observe_sequence(embeds, act_t, is_first)
            post_feats = torch.stack([s.features for s in post_states], dim=1)
            recons = self.world_model.decoder(post_feats)
            pred_rewards = self.world_model.reward_head(post_feats).squeeze(-1)

            # 1-step errors
            one_step_recon = float(torch.mean((recons[:, :-1] - obs_t[:, 1:]) ** 2).item())
            reward_target = torch.tensor(rews[:-1], dtype=torch.float32).unsqueeze(0)
            one_step_rew = float(torch.mean((pred_rewards[:, :-1] - reward_target) ** 2).item())

            # Multi-step open-loop rollouts from prefix
            horizon_errors: dict[int, float] = {}
            for H in [1, 3, 5, 10]:
                if H < T:
                    # Unroll H steps from midpoint
                    mid = T // 2
                    if mid + H < T:
                        init_s = post_states[mid]
                        cand_acts = act_t[:, mid : mid + H]
                        imagined_states = self.world_model.rssm.imagine_trajectory(init_s, cand_acts)
                        imag_feats = torch.stack([s.features for s in imagined_states], dim=1)
                        imag_obs = self.world_model.decoder(imag_feats)
                        ground_truth_obs = obs_t[:, mid + 1 : mid + H + 1]
                        err = float(torch.mean((imag_obs - ground_truth_obs) ** 2).item())
                        horizon_errors[H] = err
                    else:
                        horizon_errors[H] = one_step_recon * float(H)
                else:
                    horizon_errors[H] = one_step_recon * float(H)

        actual_return = episode.total_return
        predicted_return = float(torch.sum(pred_rewards).item())
        regret = max(0.0, predicted_return - actual_return)

        return ErrorGapAnalysisResult(
            task_name=episode.task_name,
            shift_name=shift_name,
            one_step_recon_mse=one_step_recon,
            one_step_reward_mse=one_step_rew,
            horizon_errors=horizon_errors,
            mean_predicted_return=predicted_return,
            mean_actual_return=actual_return,
            decision_regret=regret,
        )
