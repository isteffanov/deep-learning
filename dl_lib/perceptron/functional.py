from typing import Callable

import numpy as np

def initialize_weights(low: float, high: float, size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=size)

def forward(x: np.ndarray, weights: np.ndarray, bias: float = 0.0, activation: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    if activation is None:
        activation = lambda x: x
    return activation(np.dot(x, weights) + bias)