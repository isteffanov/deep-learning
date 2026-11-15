import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights, forward

def main() -> int:
    dataset = create_dataset(6)

    weights = initialize_weights(0, 10, size=1, seed=42)
    y_true = np.array([y for _, y in dataset])
    y_pred = np.array([forward(np.array([x]), weights) for x, _ in dataset])

    error = mse(y_true, y_pred)

    print(f"Dataset: {dataset}")
    print(f"Weights: {weights}")
    print(f"MSE: {error}")

    for dw in [-0.002, -0.001, 0.001, 0.002]:
        y_pred = np.array([forward(np.array([x]), weights + dw) for x, _ in dataset])
        error = mse(y_true, y_pred)

        print()
        print(f"Weights + {dw}: {weights + dw}")
        print(f"MSE: {error}")

    return 0