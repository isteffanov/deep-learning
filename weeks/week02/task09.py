"""
Create a script weeks/week2/task09.py that builds a dataset of probabilities with
a separation between classes, sweeps the threshold from low to high, computes
precision and recall at each step, prints a few sample points, and shows a plot
of both against the threshold.

Acceptance criteria:

Dataset is created via numpy.
A comparison between precision and recall against different thresholds is shown.
"""

import numpy as np
import matplotlib.pyplot as plt

from dl_lib.metrics.functional import apply_threshold, precision, recall
from weeks.week02.helpers import make_labels

def main() -> int:
    N_NEGATIVE: int = 50
    N_POSITIVE: int = 50
    SIZE: int = N_POSITIVE + N_NEGATIVE

    rng = np.random.default_rng(seed=0)
    y_true: np.ndarray = make_labels(rng, n_negative=N_NEGATIVE, n_positive=N_POSITIVE)
    probs: np.ndarray = np.where(
        y_true == 1,
        rng.beta(5, 2, SIZE),
        rng.beta(2, 5, SIZE))

    thresholds: np.ndarray = np.linspace(0, 1, 21)
    precisions: list = []
    recalls: list = []

    for threshold in thresholds:
        y_pred: np.ndarray = apply_threshold(probs, threshold)
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)


        if np.isclose(threshold, [0.2, 0.5, 0.8]).any():
            print(f"threshold={threshold}: precision={p:.4f}, recall={r:.4f}")


        precisions.append(p)
        recalls.append(r)

    plt.plot(thresholds, precisions)
    plt.plot(thresholds, recalls)

    plt.legend(["Precision", "Recall"])
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Precision / Recall Curve")
    plt.grid()

    plt.savefig("plots/week02/task09.png")

    return 0