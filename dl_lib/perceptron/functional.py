from typing import Callable

import numpy as np

h = 1.e-4

def initialize_weights(low: float, high: float, size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=size)

def forward(x: np.ndarray, weights: np.ndarray, bias: float = 0.0, activation: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    if activation is None:
        activation = lambda x: x
    return activation(np.dot(x, weights) + bias)

# also
# - Forward difference: (L(params + h) − L(params)) / h
# - Backward difference: (L(params) - L(params - h) / h
def numerical_gradient(loss_fn: Callable[[np.ndarray], float], params: np.ndarray) -> np.ndarray:
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