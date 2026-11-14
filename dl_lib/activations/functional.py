
import numpy as np

def sigmoid(array: np.ndarray) -> np.ndarray:
	return np.reciprocal(1 + np.exp(-array))

def relu(array: np.ndarray) -> np.ndarray:
	return np.maximum(0, array)

# 2 * sigmoid(2 * array) - 1 prevents overflows
def tanh(array: np.ndarray) -> np.ndarray:
	return (np.exp(array) - np.exp(-array)) / (np.exp(array) + np.exp(-array))

def softmax(array: np.ndarray) -> np.ndarray:
	# prevents overflows
    shifted = array - np.max(array, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)

def unsafe_softmax(array: np.ndarray) -> np.ndarray:
	return np.exp(array) / np.sum(np.exp(array))