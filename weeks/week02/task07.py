"""
Create a script weeks/week2/task07.py that prints all four metrics for a small balanced dataset,
then for a deliberately imbalanced one (990 negative examples, 10 positive), where a 3 of the
rare positives are missed.

Acceptance criteria:

* Calculations are done via numpy.
* Datasets are randomly generated using numpy.
* A comparison between the values of the metrics on an imbalanced dataset is shown.
"""

import numpy as np

from dl_lib.metrics.functional import accuracy, precision, recall, f1_score


def make_labels(rng: np.random.Generator, n_negative: int, n_positive: int) -> np.ndarray:
    labels = np.concatenate([np.zeros(n_negative, dtype=int), np.ones(n_positive, dtype=int)])
    return rng.permutation(labels)


def flip(rng: np.random.Generator, y: np.ndarray, candidates: np.ndarray, n: int) -> np.ndarray:
    y_flipped = y.copy()
    idx = rng.choice(candidates, size=n, replace=False)
    y_flipped[idx] = 1 - y_flipped[idx]
    return y_flipped


def print_metrics(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"{title}:")
    print(f"\tAccuracy:  {accuracy(y_true, y_pred):.3f}")
    print(f"\tPrecision: {precision(y_true, y_pred):.3f}")
    print(f"\tRecall:    {recall(y_true, y_pred):.3f}")
    print(f"\tF1 Score:  {f1_score(y_true, y_pred):.3f}")


def main() -> int:
    rng = np.random.default_rng(seed=0)

    # Balanced
    y_true_balanced = make_labels(rng, n_negative=10, n_positive=10)
    y_pred_balanced = flip(rng, y_true_balanced, np.arange(y_true_balanced.size), n=3)
    print_metrics("Balanced dataset (10 neg / 10 pos, 3 random mistakes)", y_true_balanced, y_pred_balanced)

    # Imbalanced
    y_true_imbalanced = make_labels(rng, n_negative=990, n_positive=10)
    positive_idx = np.flatnonzero(y_true_imbalanced == 1)
    y_pred_imbalanced = flip(rng, y_true_imbalanced, positive_idx, n=3)
    print_metrics("\nImbalanced dataset (990 neg / 10 pos, 3 positives missed)", y_true_imbalanced, y_pred_imbalanced)

    return 0
