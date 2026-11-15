import numpy as np

def initialize_weights(low: float, high: float, size: int) -> np.ndarray:
    return np.random.uniform(low, high, size=size)