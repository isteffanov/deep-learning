import numpy as np

EPS = 1.e-6


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    _validate(y_true, y_pred)
    
    return np.mean(np.absolute(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    _validate(y_true, y_pred)

    return np.mean(np.square(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred:np.ndarray) -> np.float32:
    _validate(y_true, y_pred)

    # double validation - not ideal
    return np.sqrt(mse(y_true, y_pred))


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if (y_true.shape != y_pred.shape):
        raise ValueError(f"mae: Cannot find the mean of arrays with different shape: {y_true.shape} and {y_pred.shape}")
    
    if (y_true.size == 0 or y_pred.size == 0):
        raise ValueError(f"mae: Cannot find the mean of empty arrays")


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    p = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    p = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean(np.sum(y_true * np.log(p), axis=-1))
