"""
Create a script weeks/week2/task08.py that builds a 990-negative/10-positive dataset, predicts
negative for every single example regardless of input, and prints its accuracy, precision, recall, and F1.

Acceptance criteria:

* The bias of accuracy is shown.
"""

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


def main() -> int:
    rng = np.random.default_rng(seed=0)

    y_true_imbalanced = make_labels(rng, n_negative=990, n_positive=10)
    y_pred_all_negative = np.zeros_like(y_true_imbalanced)

    print_metrics("Imbalanced dataset, model always predicts 0", y_true_imbalanced, y_pred_all_negative)

    return 0
