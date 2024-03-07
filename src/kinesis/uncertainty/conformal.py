"""Distribution-free conformal prediction intervals with finite-sample coverage guarantees."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ConformalInterval:
    """Conformal prediction interval for regression predictions."""
    lower: np.ndarray
    upper: np.ndarray
    point_prediction: np.ndarray
    target_coverage: float
    quantile_cutoff: float


class ConformalPredictor:
    """Computes distribution-free conformal confidence bands with exact coverage guarantees."""

    def __init__(self, significance_level: float = 0.1) -> None:
        self.alpha = float(significance_level)
        assert 0.0 < self.alpha < 1.0, f"Alpha must be in (0, 1), got {self.alpha}"
        self.calibrated = False
        self.quantile_q = 0.0

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    def calibrate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calibrate non-conformity scores on holdout calibration split."""
        preds = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(ground_truth, dtype=np.float64)
        assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"

        n = len(preds)
        assert n > 0, "Calibration dataset cannot be empty"

        # Residual magnitude non-conformity scores
        residuals = np.abs(targets - preds)
        if residuals.ndim > 1:
            # Multi-dimensional: use max norm across output dimensions
            scores = np.max(residuals, axis=-1)
        else:
            scores = residuals

        # Conformal quantile: ceiling((n + 1) * (1 - alpha)) / n
        p_val = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile_q = float(np.quantile(scores, p_val, method="higher"))
        self.calibrated = True
        return self.quantile_q

    def predict_interval(self, predictions: np.ndarray) -> ConformalInterval:
        """Compute conformal confidence intervals around point predictions."""
        assert self.calibrated, "ConformalPredictor must be calibrated before calling predict_interval"
        preds = np.asarray(predictions, dtype=np.float64)

        lower = preds - self.quantile_q
        upper = preds + self.quantile_q

        return ConformalInterval(
            lower=lower,
            upper=upper,
            point_prediction=preds,
            target_coverage=self.target_coverage,
            quantile_cutoff=self.quantile_q,
        )

    def evaluate_coverage(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute empirical coverage percentage on test data."""
        interval = self.predict_interval(predictions)
        targets = np.asarray(ground_truth, dtype=np.float64)

        if targets.ndim > 1:
            in_bounds = np.all((targets >= interval.lower) & (targets <= interval.upper), axis=-1)
        else:
            in_bounds = (targets >= interval.lower) & (targets <= interval.upper)

        empirical_coverage = float(np.mean(in_bounds))
        return empirical_coverage
