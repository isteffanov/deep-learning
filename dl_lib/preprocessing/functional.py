import math
import numpy as np

def min_max_normalize(array: np.ndarray) -> np.ndarray:
    if len(array) == 0:
        return []
    
    min: float = array.min()
    max: float = array.max()

    print(min, max)

    if math.isclose(min, max):
        return np.zeros(array.shape)

    return (array - min) / (max - min)