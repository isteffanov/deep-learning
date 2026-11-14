"""
Create a script weeks/week2/task03.py that prints mse on Task 01's exact numbers, then replaces
one prediction with an extreme outlier and prints how much mae and mse each change.

Lastly, show how PyTorch can also be used to calculate mse via torch.nn.functional.mse_loss.

Acceptance criteria:

* Calculations are done via numpy.
* PyTorch is used in the script to show the built-in way of calculating mse.
* Comparison is shown between mae and mse.
"""

import torch
import numpy as np

from dl_lib.losses.functional import mae, mse

def main() -> int:
    y_true: np.ndarray = np.array([5, 6, 3, 4], dtype=np.float32)
    y_pred: np.ndarray = np.array([7, 5, 4, 1], dtype=np.float32)

    print(f"y_true:\t {y_true.tolist()}")
    print(f"y_pred:\t {y_pred.tolist()}")
    print(f"mse:\t {mse(y_true, y_pred)}")

    y_pred_outlier: np.ndarray = np.array([7, 5, 4, 100], dtype=np.float32)
    print(f"y_pred_with_outlier:\t {y_pred_outlier.tolist()}")
    print(f"mae with outlier:\t {mae(y_true, y_pred_outlier)}")
    print(f"mse with outlier:\t {mse(y_true, y_pred_outlier)}")

    torch.nn.functional.mse_loss(torch.from_numpy(y_pred), torch.from_numpy(y_true)) # swapped

    return 0