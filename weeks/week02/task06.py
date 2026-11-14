"""
Create a script weeks/week2/task06.py that builds a confusion matrix from a small
hand-picked set of true labels and predictions and prints all four counts.

Acceptance criteria:

* Calculations are done via numpy.
"""

import numpy as np

from dl_lib.metrics.functional import confusion_matrix

def main() -> int:
    y_true: np.ndarray = np.array([1, 0, 1, 0, 1])
    y_pred: np.ndarray = np.array([1, 0, 0, 1, 1])

    cm: dict[str, int] = confusion_matrix(y_true, y_pred)

    print(f"True Positives:  {cm['tp']}")
    print(f"True Negatives:  {cm['tn']}")
    print(f"False Positives: {cm['fp']}")
    print(f"False Negatives: {cm['fn']}")

    return 0