import numpy as np

from dl_lib.metrics.functional import accuracy, precision, recall, f1_score

def make_labels(rng: np.random.Generator, n_negative: int, n_positive: int) -> np.ndarray:
    labels = np.concatenate([np.zeros(n_negative, dtype=int), np.ones(n_positive, dtype=int)])
    return rng.permutation(labels)


def print_metrics(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"{title}:")
    print(f"\tAccuracy:  {accuracy(y_true, y_pred):.3f}")
    print(f"\tPrecision: {precision(y_true, y_pred):.3f}")
    print(f"\tRecall:    {recall(y_true, y_pred):.3f}")
    print(f"\tF1 Score:  {f1_score(y_true, y_pred):.3f}")