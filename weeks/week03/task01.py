import numpy as np

from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights

def main() -> int:
    dataset: list[tuple[int, int]] = create_dataset(4)
    weights_1: np.ndarray = initialize_weights(0, 10, 1)
    weights_2: np.ndarray = initialize_weights(0, 100, 1)

    print(f"dataset: {dataset}")
    print(f"initialize_weights(0, 10, 1): {weights_1[0]}")
    print(f"initialize_weights(0, 100, 1): {weights_2[0]}")

    return 0