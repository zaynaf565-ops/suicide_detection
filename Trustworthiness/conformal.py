"""
Conformal prediction for the binary task.

Conformal prediction turns a single prediction into a set of labels that is
guaranteed, on average, to contain the true label at least a chosen fraction of
the time. Here the target is ninety percent coverage. For this binary task a
prediction set can hold one label, both labels or neither. When it does not hold
exactly one label the model is treated as abstaining, which is the signal to
send that post to a human reviewer.

The non-conformity score is the least ambiguous set valued classifier score
(Sadinle et al., 2019), which for the binary case is one minus the probability
of the true class. The threshold is calibrated on the held out calibration set,
which is why that set was kept apart from training and tuning. For two classes
this is the adaptive prediction set rule in its simplest and most stable form,
so the regularisation term in RAPS has no effect and is not needed here.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def lac_scores(p1, y):
    """One minus the probability assigned to the true class."""
    p1 = np.clip(p1, 1e-7, 1 - 1e-7)
    p_true = np.where(y == 1, p1, 1 - p1)
    return 1.0 - p_true


def calibrate_qhat(p1_cal, y_cal, alpha):
    s = lac_scores(p1_cal, y_cal)
    n = len(s)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(s, level, method="higher"))


def build_sets(p1, qhat):
    """Include class c when one minus its probability is within qhat."""
    p1 = np.clip(p1, 1e-7, 1 - 1e-7)
    inc1 = (1 - p1) <= qhat           # include class 1
    inc0 = p1 <= qhat                 # include class 0
    sets = []
    for a0, a1 in zip(inc0, inc1):
        s = []
        if a0: s.append(0)
        if a1: s.append(1)
        sets.append(tuple(s))
    return sets


def evaluate_sets(sets, y):
    sizes = np.array([len(s) for s in sets])
    covered = np.array([y[i] in sets[i] for i in range(len(y))])
    return {
        "coverage":     round(float(covered.mean()), 4),
        "avg_set_size": round(float(sizes.mean()), 4),
        "abstention":   round(float((sizes != 1).mean()), 4),
        "singleton":    round(float((sizes == 1).mean()), 4),
    }


def run_conformal(y_cal, p_cal, y_test, p_test, alpha=0.10, label="model"):
    """Calibrate the threshold on the calibration split, evaluate on test."""
    qhat = calibrate_qhat(p_cal, y_cal, alpha)
    sets_test = build_sets(p_test, qhat)
    ev = evaluate_sets(sets_test, y_test)
    ev.update({"model": label, "qhat": round(qhat, 4), "alpha": alpha})
    return ev, sets_test


def plot_abstention_by_confidence(p_test, sets_test, label, path, n_bins=10):
    conf = np.maximum(p_test, 1 - p_test)
    sizes = np.array([len(s) for s in sets_test])
    abstain = (sizes != 1).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    centres, rates = [], []
    for i in range(n_bins):
        hi = conf <= bins[i + 1] if i == n_bins - 1 else conf < bins[i + 1]
        m = (conf >= bins[i]) & hi
        if m.sum():
            centres.append((bins[i] + bins[i + 1]) / 2)
            rates.append(abstain[m].mean())
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(centres, rates, width=0.045, color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Model confidence"); ax.set_ylabel("Abstention rate")
    ax.set_title(f"Abstention by confidence, {label}")
    ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
