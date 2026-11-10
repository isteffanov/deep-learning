"""
Create a script weeks/week2/task01.py that prints y_true, y_pred, the elementwise errors, absolute errors, and squared errors, 
then the mean error, the mean absolute error, the mean squared error, and its square root.

Acceptance criteria:

* Calculations are done via numpy.
"""

import numpy as np

def main() -> int:

    y_true: np.ndarray = np.array([5, 6, 3, 4])
    y_pred: np.ndarray = np.array([7, 5, 4, 1])

    print(f"y_true:\t {y_true.tolist()}")
    print(f"y_pred:\t {y_pred.tolist()}")
    print(f"errors:\t {(y_pred - y_true).tolist()}")
    print(f"absolute: {np.absolute(y_pred - y_true).tolist()}")
    print()

    print(f"mean error: {(y_pred - y_true).mean()}")
    print(f"mae:\t {np.absolute(y_pred - y_true).mean()}")
    print(f"mse:\t {np.square(y_pred - y_true).mean()}")
    print(f"rmse:\t {np.sqrt(np.square(y_pred - y_true).mean())}")

    return 0