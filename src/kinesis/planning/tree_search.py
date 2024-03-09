"""Latent Monte Carlo Tree Search (MCTS) with PUCT selection and learned world model expansion."""

from dataclasses import dataclass, field
import numpy as np
import torch

from kinesis.contracts.action import ActionSpec
from kinesis.models.rssm import RSSMState
from kinesis.models.world_model import WorldModel


@dataclass
class MCTSNode:
    """Node in latent Monte Carlo search tree."""
    state: RSSMState
    parent: "MCTSNode | None" = None
    action_from_parent: np.ndarray | None = None
    reward_from_parent: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visit_counts: dict[int, int] = field(default_factory=dict)
    total_values: dict[int, float] = field(default_factory=dict)
    priors: dict[int, float] = field(default_factory=dict)


@dataclass
class MCTSResult:
    """Output from latent Monte Carlo tree search."""
    best_action: np.ndarray
    best_action_index: int
    root_value: float
    total_simulations: int
    visit_distribution: np.ndarray


class LatentMCTSPlanner:
    """Monte Carlo Tree Search over learned latent world model dynamics."""

    def __init__(
        self,
        world_model: WorldModel,
        action_spec: ActionSpec,
        action_candidates: list[np.ndarray] | None = None,
        num_simulations: int = 50,
        c_puct: float = 1.414,
        discount: float = 0.99,
    ) -> None:
        self.world_model = world_model
        self.action_spec = action_spec
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.discount = discount

        if action_candidates is None:
            # Generate default discretized action candidates spanning limits
            self.actions = self._generate_default_candidates()
        else:
            self.actions = [np.asarray(a, dtype=np.float32) for a in action_candidates]

        self.num_actions = len(self.actions)

    def _generate_default_candidates(self) -> list[np.ndarray]:
        """Generate default discrete action grid across action bounds."""
        dim = self.action_spec.dim
        low = self.action_spec.minimum
        high = self.action_spec.maximum

        candidates = [np.zeros(dim, dtype=np.float32)]
        for d in range(dim):
            act_pos = np.zeros(dim, dtype=np.float32)
            act_pos[d] = float(high[d] * 0.8)
            candidates.append(act_pos)

            act_neg = np.zeros(dim, dtype=np.float32)
            act_neg[d] = float(low[d] * 0.8)
            candidates.append(act_neg)

        return candidates

    def _select_child_action(self, node: MCTSNode) -> int:
        """Select action maximizing PUCT upper confidence bound."""
        total_visits = sum(node.visit_counts.values())
        best_score = -float("inf")
        best_a = 0

        for a_idx in range(self.num_actions):
            n_a = node.visit_counts.get(a_idx, 0)
            q_a = (node.total_values.get(a_idx, 0.0) / n_a) if n_a > 0 else 0.0
            p_a = node.priors.get(a_idx, 1.0 / self.num_actions)

            # PUCT formula
            u_a = self.c_puct * p_a * (np.sqrt(total_visits + 1e-8) / (1 + n_a))
            score = q_a + u_a

            if score > best_score:
                best_score = score
                best_a = a_idx

        return best_a

    def plan(self, initial_state: RSSMState) -> MCTSResult:
        """Execute MCTS search from current latent belief state."""
        device = initial_state.deter.device
        root = MCTSNode(state=initial_state)

        # Initialize root action priors uniformly
        for a_idx in range(self.num_actions):
            root.priors[a_idx] = 1.0 / self.num_actions
            root.visit_counts[a_idx] = 0
            root.total_values[a_idx] = 0.0

        with torch.no_grad():
            for _ in range(self.num_simulations):
                node = root
                search_path = [node]
                actions_taken = []

                # 1. Selection
                while actions_taken and (actions_taken[-1] in node.children):
                    a_idx = self._select_child_action(node)
                    actions_taken.append(a_idx)
                    node = node.children[a_idx]
                    search_path.append(node)

                # Select action at current leaf
                a_idx = self._select_child_action(node)
                actions_taken.append(a_idx)

                # 2. Expansion using RSSM imagination step
                act_tensor = torch.from_numpy(self.actions[a_idx]).float().unsqueeze(0).to(device)
                next_state = self.world_model.rssm.imagine_step(node.state, act_tensor, sample=False)

                feat = torch.cat([next_state.deter, next_state.stoch], dim=-1)
                pred_reward = float(self.world_model.reward_head(feat).item())

                # Value approximation (leaf evaluation)
                value = pred_reward

                # Create child
                child = MCTSNode(
                    state=next_state,
                    parent=node,
                    action_from_parent=self.actions[a_idx],
                    reward_from_parent=pred_reward,
                )
                for next_a in range(self.num_actions):
                    child.priors[next_a] = 1.0 / self.num_actions
                    child.visit_counts[next_a] = 0
                    child.total_values[next_a] = 0.0

                node.children[a_idx] = child

                # 3. Backpropagation along search path
                curr_value = value
                for parent_node, a_taken in zip(reversed(search_path), reversed(actions_taken)):
                    parent_node.visit_counts[a_taken] += 1
                    parent_node.total_values[a_taken] += curr_value
                    curr_value = parent_node.reward_from_parent + self.discount * curr_value

        # Select action with highest visit count at root
        visits = np.array([root.visit_counts[a] for a in range(self.num_actions)], dtype=np.float32)
        best_a = int(np.argmax(visits))
        visit_dist = visits / np.sum(visits)

        mean_val = root.total_values[best_a] / max(root.visit_counts[best_a], 1)

        return MCTSResult(
            best_action=self.actions[best_a],
            best_action_index=best_a,
            root_value=float(mean_val),
            total_simulations=self.num_simulations,
            visit_distribution=visit_dist,
        )
