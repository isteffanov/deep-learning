import numpy as np


def apply_threshold(probs: np.ndarray, threshold: np.float32) -> np.ndarray:
    if (any(map(lambda x: x > 1 or x < 0, probs))):
        raise ValueError(f"apply_threshold: Accepts only array of probabilities between 0 and 1, got {probs}")

    if (threshold > 1 or threshold < 0):
        raise ValueError(f"appy_threshold: Accepts threshold as a probability between 0 and 1, got {threshold}")

    # i would usually vectorize  
    return np.array([1 if x >= threshold else 0 for x in probs])