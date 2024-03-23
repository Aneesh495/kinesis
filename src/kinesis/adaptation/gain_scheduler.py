"""Robust gain scheduler interpolating policy parameters across identified physical regimes."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OperatingRegime:
    """Declared physical operating point with associated feedback gains."""
    name: str
    parameter_vector: np.ndarray  # Physical parameter signature [d]
    feedback_gain: np.ndarray     # Optimal gain matrix for this regime [m, n]
    uncertainty_weight: float = 1.0


@dataclass
class ScheduledGainResult:
    """Output from gain interpolation."""
    interpolated_gain: np.ndarray
    regime_weights: dict[str, float]
    dominant_regime: str
    uncertainty_weight: float


class RobustGainScheduler:
    """Smooth convex gain scheduler blending control gains based on estimated physical regime."""

    def __init__(
        self,
        regimes: list[OperatingRegime],
        kernel_bandwidth: float = 0.5,
    ) -> None:
        assert len(regimes) > 0, "At least one operating regime must be registered"
        self.regimes = regimes
        self.bandwidth = float(kernel_bandwidth)

        # Dimension validation
        self.param_dim = len(regimes[0].parameter_vector)
        self.gain_shape = regimes[0].feedback_gain.shape
        for r in regimes:
            assert len(r.parameter_vector) == self.param_dim, "Inconsistent parameter vector dimension"
            assert r.feedback_gain.shape == self.gain_shape, "Inconsistent feedback gain shape"

    def schedule(self, estimated_parameters: np.ndarray) -> ScheduledGainResult:
        """Compute convex combination of feedback gains based on Gaussian radial basis functions."""
        theta = np.asarray(estimated_parameters, dtype=np.float64)
        assert theta.shape == (self.param_dim,), f"Shape mismatch: {theta.shape} vs {(self.param_dim,)}"

        # Compute RBF distances
        distances = []
        for r in self.regimes:
            dist_sq = float(np.sum((theta - r.parameter_vector)**2))
            distances.append(dist_sq)

        dist_arr = np.array(distances, dtype=np.float64)
        logits = -dist_arr / (2.0 * self.bandwidth**2)
        max_logit = np.max(logits)
        exp_vals = np.exp(logits - max_logit)
        weights = exp_vals / np.sum(exp_vals)

        # Blend gains and uncertainty weights
        blended_gain = np.zeros(self.gain_shape, dtype=np.float64)
        blended_unc = 0.0
        weight_dict = {}

        for w_val, regime in zip(weights, self.regimes):
            blended_gain += w_val * regime.feedback_gain
            blended_unc += w_val * regime.uncertainty_weight
            weight_dict[regime.name] = float(w_val)

        dom_idx = int(np.argmax(weights))
        dominant = self.regimes[dom_idx].name

        return ScheduledGainResult(
            interpolated_gain=blended_gain,
            regime_weights=weight_dict,
            dominant_regime=dominant,
            uncertainty_weight=blended_unc,
        )
