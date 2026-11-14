import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    _validate(y_true, y_pred)
    
    return np.mean(np.absolute(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    _validate(y_true, y_pred)

    return np.mean(np.square(y_true - y_pred))


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if (y_true.shape != y_pred.shape):
        raise ValueError(f"mae: Cannot find the mean of arrays with different shape: {y_true.shape} and {y_pred.shape}")
    
    if (y_true.size == 0 or y_pred.size == 0):
        raise ValueError(f"mae: Cannot find the mean of empty arrays")

