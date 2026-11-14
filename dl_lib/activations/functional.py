
import numpy as np

def sigmoid(array: np.ndarray) -> np.ndarray:
	return np.reciprocal(1 + np.exp(-array))

def relu(array: np.ndarray) -> np.ndarray:
	return np.maximum(0, array)

def tanh(array: np.ndarray) -> np.ndarray:
	return (np.exp(array) - np.exp(-array)) / (np.exp(array) + np.exp(-array))