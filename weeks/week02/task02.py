"""
Description:

Define mae in dl_lib.losses.functional. Takes the true values and the predicted values as 
same-shaped arrays; returns the mean absolute error between them as a float.

Create a script weeks/week2/task02.py that calls mae on Task 01's exact numbers and prints
the result. It should then also run torch.nn.functional.l1_loss on the same inputs to demonstrate
how PyTorch can be used to do this on the fly.

Acceptance criteria:

* Calculations are done via numpy.
* PyTorch is used in the script to show the built-in way of calculating mae.
"""

import torch
import numpy as np

from dl_lib.losses.functional import mae

def main() -> int:
    y_true: np.ndarray = np.array([5, 6, 3, 4], dtype=np.float32)
    y_pred: np.ndarray = np.array([7, 5, 4, 1], dtype=np.float32)

    print(f"y_true: {y_true.tolist()}")
    print(f"y_pred: {y_pred.tolist()}")
    print(f"MAE: {mae(y_true, y_pred)}")

    torch.nn.functional.l1_loss(torch.from_numpy(y_pred), torch.from_numpy(y_true)) # swapped

    return 0