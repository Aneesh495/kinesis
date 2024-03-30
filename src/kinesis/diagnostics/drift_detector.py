"""Sequential change-point and concept drift detection via Page-Hinkley and CUSUM algorithms."""

from dataclasses import dataclass
import numpy as np


@dataclass
class DriftDetectionResult:
    """Output from sequential drift detector."""
    drift_detected: bool
    test_statistic: float
    threshold: float
    current_mean: float
    samples_seen: int


class PageHinkleyDriftDetector:
    """Page-Hinkley test for detecting abrupt changes in streaming error residual distributions."""

    def __init__(self, delta: float = 0.05, threshold: float = 10.0, alpha_forget: float = 1.0) -> None:
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha_forget)

        self.n = 0
        self.running_mean = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0

    def reset(self) -> None:
        """Reset internal detector statistics."""
        self.n = 0
        self.running_mean = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0

    def update(self, sample: float) -> DriftDetectionResult:
        """Update detector with next scalar sample and check for change point."""
        x = float(sample)
        self.n += 1

        # Welford running mean
        self.running_mean += (x - self.running_mean) / self.n

        # Update cumulative sum: U_t = sum(x_i - mean - delta)
        self.cum_sum += (x - self.running_mean - self.delta)
        if self.cum_sum < self.min_cum_sum:
            self.min_cum_sum = self.cum_sum

        # Page-Hinkley statistic: PH_t = U_t - min_U
        ph_stat = self.cum_sum - self.min_cum_sum
        drift = bool(ph_stat > self.threshold)

        if drift:
            self.reset()

        return DriftDetectionResult(
            drift_detected=drift,
            test_statistic=float(ph_stat),
            threshold=self.threshold,
            current_mean=self.running_mean,
            samples_seen=self.n,
        )


class CUSUMDriftDetector:
    """Two-sided Cumulative Sum (CUSUM) change-point detector."""

    def __init__(self, target_mean: float = 0.0, slack_k: float = 0.5, threshold_h: float = 5.0) -> None:
        self.mu0 = float(target_mean)
        self.k = float(slack_k)
        self.h = float(threshold_h)

        self.s_pos = 0.0
        self.s_neg = 0.0
        self.n = 0

    def reset(self) -> None:
        """Reset CUSUM statistics."""
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.n = 0

    def update(self, sample: float) -> DriftDetectionResult:
        """Update two-sided CUSUM statistic."""
        x = float(sample)
        self.n += 1

        # S^+ = max(0, S^+ + x - mu0 - k)
        self.s_pos = max(0.0, self.s_pos + x - self.mu0 - self.k)
        # S^- = max(0, S^- - x + mu0 - k)
        self.s_neg = max(0.0, self.s_neg - x + self.mu0 - self.k)

        stat = max(self.s_pos, self.s_neg)
        drift = bool(stat > self.h)

        if drift:
            self.reset()

        return DriftDetectionResult(
            drift_detected=drift,
            test_statistic=float(stat),
            threshold=self.h,
            current_mean=x,
            samples_seen=self.n,
        )
