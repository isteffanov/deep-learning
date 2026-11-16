from typing import Callable

import numpy as np

from dl_lib.losses.functional import mse

h = 1.e-4

def initialize_weights(low: float, high: float, size: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=size)

def forward(x: np.ndarray, weights: np.ndarray, bias: float = 0.0, activation: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    if activation is None:
        activation = lambda x: x
    return activation(np.dot(x, weights) + bias)

# also
# - Forward difference: (L(params + h) − L(params)) / h
# - Backward difference: (L(params) - L(params - h) / h
def numerical_gradient(loss_fn: Callable[[np.ndarray], np.float32], params: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(params)
    plus = params.copy()
    minus = params.copy()

    for idx in np.ndindex(params.shape):
        plus[idx] += h
        minus[idx] -= h

        grad[idx] = (loss_fn(plus) - loss_fn(minus)) / (2 * h)

        plus[idx] -= h
        minus[idx] += h

    return grad

def gradient_descent_step(params: np.ndarray, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
    return params - learning_rate * gradient

def dataset_loss(features: np.ndarray, targets, weights, bias = 0.0, activation = None, loss = mse) -> np.float32:
    y_pred = np.array([forward(weights, x, bias, activation) for x in features]).reshape(targets.shape)
    return loss(targets, y_pred)

def dataset_loss_given_bias(features, targets, weights, bias, activation = None, loss = mse) -> np.float32:
    return dataset_loss(features, targets, weights, bias, activation, loss)

def analytical_gradient(dataset, weights, bias = 0.0, activation = None, activation_derivative = None, loss = "mse") -> tuple[np.ndarray, np.float32]:
    if activation is None:
        activation = lambda x: x

    if activation_derivative is None:
        activation_derivative = lambda x: 1

    features = np.array([x for x, _ in dataset])
    targets = np.array([y for _, y in dataset])
    N = len(dataset)

    z = features @ weights + bias  # shape (N,)
    y_pred = activation(z)
    if loss == "mse":
        delta = 2 * (y_pred - targets) * activation_derivative(z)  # dL/dŷ × f'(z)
    elif loss == "bce":
        delta = y_pred - targets  # sigmoid's f'(z) cancels with BCE's dL/dŷ
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    weight_grad = features.T @ delta / N  # × x_i, averaged
    bias_grad = np.sum(delta) / N  # × 1, averaged

    return weight_grad, bias_grad