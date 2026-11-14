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

from dl_lib.metrics.functional import accuracy, precision, recall, f1_score
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

    print(probs)

    return 0