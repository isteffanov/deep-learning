"""
Create a script weeks/week2/task05.py that applies apply_threshold to a fixed set
of probabilities ([0.1, 0.8, 0.4, 0.7, 0.9]) at several thresholds from 0.1 to 0.9
and prints the resulting predictions at each one.

Acceptance criteria:

* Calculations are done via numpy.
"""

import numpy as np

from dl_lib.metrics.functional import apply_threshold

def main() -> int:
    probs: np.ndarray = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
    thresholds: list = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"probababilities: {probs.tolist()}")
    print()

    for threshold in thresholds:
        print(f"threshold: {threshold}, predictions: {apply_threshold(probs, threshold)}")

    return 0