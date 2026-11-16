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

    print(f"Loss: {loss(weights)}")
    print(f"Grad: {numerical_gradient(loss, weights)}")
    print()
    w1 = weights - numerical_gradient(loss, weights)
    l1 = loss(w1)

    w2 = gradient_descent_step(weights, numerical_gradient(loss, weights), 0.001)
    l2 = loss(w2)
    print(f"weights - raw gradient: weights={w1}, loss={l1}")
    print(f"weights - GD, lr=0.001: weights={w2}, loss={l2}")
    print()

    for epoch in range(10):
        gradient = numerical_gradient(loss, weights)
        weights = gradient_descent_step(weights, gradient, 0.001)
        cur_loss = loss(weights)

        print(f"epoch: {epoch + 1:2d}, weights: {weights[0]:4f}, loss: {cur_loss:4f}")

    return 0