import numpy as np

from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights, dataset_loss, forward, numerical_gradient, analytical_gradient

from dl_lib.losses.functional import mse


def main() -> int:
    dataset = create_dataset(6)

    weights = initialize_weights(0, 10, size=1, seed=42)
    X = np.array([x for x, _ in dataset])
    y_true = np.array([y for _, y in dataset])

    def loss(ws: np.ndarray) -> np.float32:
        y_pred = np.array([forward(ws, x) for x in X]).reshape(y_true.shape)
        return mse(y_true, y_pred)

    num_grad = numerical_gradient(loss, weights)[0]
    anl_grad = analytical_gradient(dataset, weights)[0][0]

    print(f"Loss: {dataset_loss(X, y_true, weights):4f}")
    print(f"Numerical gradient:  {num_grad:4f}")
    print(f"Analytical gradient: {anl_grad:4f}")
    print(f"Difference: {np.abs(anl_grad - num_grad)}")


    return 0