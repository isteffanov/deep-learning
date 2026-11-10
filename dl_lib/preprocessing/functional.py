import math
import numpy as np

def min_max_normalize(array: np.ndarray) -> np.ndarray:
    if array.ndim != 1:
        raise ValueError(f"expected a 1D array, got shape {array.shape}")

    if len(array) == 0:
        return np.array([])
    
    min: float = array.min()
    max: float = array.max()

    if math.isclose(min, max):
        return np.zeros(array.shape)

    return (array - min) / (max - min)

def standardize(array: np.ndarray) -> np.ndarray:
    return np.array([])