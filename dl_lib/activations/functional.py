
import numpy as np

def sigmoid(array: np.ndarray) -> np.ndarray:
	return np.reciprocal(1 + np.exp(-array))
