import numpy as np


def apply_threshold(probs: np.ndarray, threshold: np.float32) -> np.ndarray:
    if (any(map(lambda x: x > 1 or x < 0, probs))):
        raise ValueError(f"apply_threshold: Accepts only array of probabilities between 0 and 1, got {probs}")

    if (threshold > 1 or threshold < 0):
        raise ValueError(f"appy_threshold: Accepts threshold as a probability between 0 and 1, got {threshold}")

    # i would usually vectorize  
    return np.array([1 if x >= threshold else 0 for x in probs])

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    if y_true.dtype != np.bool:
        y_true = np.array(y_true, dtype=np.bool)
    if y_pred.dtype != np.bool:
        y_pred = np.array(y_pred, dtype=np.bool)

    return {
         "tp": np.sum(y_true & y_pred),
         "tn": np.sum(np.logical_not(y_true) & np.logical_not(y_pred)),
         "fp": np.sum(np.logical_not(y_true) & y_pred),
         "fn": np.sum(y_true & np.logical_not(y_pred))
    }