import numpy as np

from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights, forward, numerical_gradient, gradient_descent_step
from dl_lib.losses.functional import mse

def main() -> int:
    dataset = create_dataset(6)

    weights = initialize_weights(0, 10, size=1, seed=42)
    X = np.array([x for x, _ in dataset])
    y_true = np.array([y for _, y in dataset])

    def loss(ws: np.ndarray) -> np.float32:
        y_pred = np.array([forward(ws, x) for x in X]).reshape(y_true.shape)
        return mse(y_true, y_pred)

    print(f"Seed=42, initial weights={weights[0]:4f}")
    for epoch in range(500):
        gradient = numerical_gradient(loss, weights)
        weights = gradient_descent_step(weights, gradient, 0.001)
        cur_loss = loss(weights)

    print(f"After 500 epochs, weights: {weights[0]:4f}, loss: {cur_loss:4f}")

    weights = initialize_weights(0, 10, size=1)
    print(f"Training without seed, initial weights: {weights[0]:4f}")
    for epoch in range(500):
        gradient = numerical_gradient(loss, weights)
        weights = gradient_descent_step(weights, gradient, 0.001)
        cur_loss = loss(weights)
    print(f"After 500 epochs, weights: {weights[0]:4f}, loss: {cur_loss:4f}")

    return 0