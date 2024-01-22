"""Actual versus imagined trajectory visualizer and terminal inspector."""

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from kinesis.contracts import Episode


class TrajectoryViewer:
    """Renders actual execution traces alongside imagined candidate rollouts."""

    @staticmethod
    def render_ascii_summary(episode: Episode) -> str:
        """Generate formatted terminal report of episode execution."""
        lines = [
            f"=== Episode Inspection Summary: {episode.episode_id} ===",
            f"Task: {episode.task_name} | Seed: {episode.seed} | Shift: {episode.shift_category.value}",
            f"Transitions: {len(episode.transitions)} | Termination: {episode.termination_reason.value}",
            f"Cumulative Return: {episode.total_return:.3f}",
            f"Total Constraint Cost: {episode.total_constraint_cost:.3f}",
            f"Simulator Substeps: {episode.total_simulator_substeps} | Duration: {episode.duration_seconds:.2f}s",
            "----------------------------------------------------------------",
        ]

        # Action statistics
        actions = np.array([t.action for t in episode.transitions])
        act_norms = np.linalg.norm(actions, axis=-1)
        lines.append(f"Mean Action Magnitude: {np.mean(act_norms):.3f} (Max: {np.max(act_norms):.3f})")
        lines.append("----------------------------------------------------------------")

        return "\n".join(lines)

    @staticmethod
    def render_diagnostic_plot(
        episode: Episode,
        output_path: Path | str,
        imagined_rewards: list[float] | None = None,
    ) -> Path:
        """Render multi-panel diagnostic chart comparing execution dynamics and predictions."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        steps = [t.step_index for t in episode.transitions]
        rewards = [t.reward for t in episode.transitions]
        costs = [t.constraint_cost for t in episode.transitions]
        actions = np.array([t.action for t in episode.transitions])

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Panel 1: Rewards (Actual vs Imagined)
        axs[0].plot(steps, rewards, label="Actual Real Reward", color="blue", linewidth=1.5)
        if imagined_rewards is not None:
            img_len = min(len(steps), len(imagined_rewards))
            axs[0].plot(steps[:img_len], imagined_rewards[:img_len], label="Imagined Prior Reward", color="orange", linestyle="--")
        axs[0].set_ylabel("Reward")
        axs[0].set_title(f"Episode Trajectory Audit: {episode.task_name} ({episode.shift_category.value})")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        # Panel 2: Actions
        for a_dim in range(actions.shape[1]):
            axs[1].plot(steps, actions[:, a_dim], label=f"Action dim {a_dim}")
        axs[1].set_ylabel("Applied Action")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="upper right")

        # Panel 3: Constraint Violations
        axs[2].plot(steps, costs, label="Constraint Cost", color="red")
        axs[2].set_ylabel("Cost")
        axs[2].set_xlabel("Environment Steps")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close(fig)

        return out
