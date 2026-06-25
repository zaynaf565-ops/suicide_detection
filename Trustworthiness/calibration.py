"""
Calibration.

A model can be accurate yet still report dishonest probabilities. Calibration
measures and corrects that. Two numbers summarise it. The expected calibration
error is the average gap between how confident the model is and how often it is
right, across probability bins. The maximum calibration error is the worst gap
in any single bin.

Temperature scaling (Guo et al., 2017) is the correction used here. It fits a
single scalar that divides the logits before the sigmoid, learned on the
calibration set by minimising the negative log likelihood. It cannot change the
ranking of predictions, so it never changes accuracy, it only adjusts how
confident the probabilities are.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.calibration import calibration_curve

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins, ece, N = np.linspace(0, 1, n_bins + 1), 0.0, len(y_true)
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.sum():
            ece += (m.sum() / N) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def maximum_calibration_error(y_true, y_prob, n_bins=10):
    bins, mce = np.linspace(0, 1, n_bins + 1), 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.sum():
            mce = max(mce, abs(y_true[m].mean() - y_prob[m].mean()))
    return float(mce)


class TemperatureScaler:
    """Single scalar temperature, fit by minimising NLL on the calibration set."""

    def __init__(self):
        self.T = 1.0

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def fit(self, y_prob, y_true):
        z = self._logit(y_prob)

        def nll(T):
            p = np.clip(expit(z / T), 1e-7, 1 - 1e-7)
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        self.T = float(res.x)
        return self

    def transform(self, y_prob):
        return expit(self._logit(y_prob) / self.T)


def reliability_diagram(y_true, p_before, p_after, label, path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1, label="Perfect")
    for p, name, col in [(p_before, "Before", "#c44e52"), (p_after, "After", "#4c72b0")]:
        frac, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac, marker="o", lw=1.4, color=col, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability, {label}")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def calibrate_model(y_cal, p_cal, y_test, p_test, label, n_bins=10, plot_path=None):
    """Fit temperature on calibration probs, apply to test, report before and after.

    Returns the metrics, the calibrated test probabilities and the calibrated
    calibration probabilities. The last is needed so the conformal stage uses the
    same transform on both splits.
    """
    scaler = TemperatureScaler().fit(p_cal, y_cal)
    p_test_cal = scaler.transform(p_test)
    p_cal_cal = scaler.transform(p_cal)
    out = {
        "model": label,
        "temperature": round(scaler.T, 4),
        "ece_before": round(expected_calibration_error(y_test, p_test, n_bins), 4),
        "ece_after":  round(expected_calibration_error(y_test, p_test_cal, n_bins), 4),
        "mce_before": round(maximum_calibration_error(y_test, p_test, n_bins), 4),
        "mce_after":  round(maximum_calibration_error(y_test, p_test_cal, n_bins), 4),
    }
    if plot_path is not None:
        reliability_diagram(y_test, p_test, p_test_cal, label, plot_path, n_bins)
    return out, p_test_cal, p_cal_cal
