"""
Create a script weeks/week2/task04.py that holds three prediction errors fixed and grows a 
fourth from 0 to 10, computing mae, mse, and rmse at each step, printing the three values
at a few sample points, and showing a plot of all three against the growing error.

Acceptance criteria:

* Calculations are done via numpy.
* The rate of growth of each of the three loss functions is shown.
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from dl_lib.losses.functional import mae, mse, rmse

def main() -> int:

    y_true: np.ndarray = np.array([1, 2, 3, 4])
    y_pred: np.ndarray = np.array([0, 1.5, 4, 4])

    mutated_prediction: Callable[[int], np.array] = lambda i: y_pred + np.array([0, 0, 0, i])

    x_points = np.linspace(0, 10, 10_000)
    mae_points = np.array([mae(y_true, mutated_prediction(i)) for i in x_points])
    mse_points = np.array([mse(y_true, mutated_prediction(i)) for i in x_points])
    rmse_points = np.array([rmse(y_true, mutated_prediction(i)) for i in x_points])

    plt.plot(x_points, mae_points)
    plt.plot(x_points, mse_points)
    plt.plot(x_points, rmse_points)

    plt.grid()
    plt.legend(["MAE", "MSE", "RMSE"])
    plt.xlabel("the growing error, having the other ones fixed")
    plt.ylabel("loss value")
    plt.title("MAE, MSE, RMSE as one prediction error grows")

    plt.savefig("plots/week02/task04.png")

    return 0