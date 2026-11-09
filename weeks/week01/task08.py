"""
Description:

Let's compute the same sum two ways on a large array and time both.

Create a script weeks/week01/task08.py that:

* Generates a large 1D array of random values.
* Sums it with an explicit Python for loop.
* Sums it again with NumPy's vectorized method.
* Times both and prints the two sums, their timings, and the speedup. It then prints the mean, standard deviation, min, 
and max of the array, computed with NumPy's vectorized methods.

Acceptance criteria:

* The array contains random values.
* Times for for loop are compared to times of vectorized numpy methods.
"""

import time
import numpy as np

from typing import Callable


def main() -> int:
    array: np.ndarray = np.random.rand(1_000_00)
    print(f"Array length: {array.size}")

    print("Standard summation:")
    time_execution(standard_summation(array))

    print("Vectorized summation:")
    time_execution(vectorized_summation(array))

    print(f"Vectorized mean: {array.mean()}")
    print(f"Vectorized std: {array.std()}")
    print(f"Vectorized min: {array.min()}")
    print(f"Vectorized max {array.max()}")

    return 0

def standard_summation(array: np.ndarray) -> Callable[[], int]:
    def curry() -> int:
        sum: int = 0
        for el in array:
            sum += el

        return sum

    return curry

def vectorized_summation(array: np.ndarray) -> Callable[[], int]:
    def curry() -> int:
        return array.sum()

    return curry

def time_execution(executable: Callable[[], int]) -> None:
    start = time.time()
    total = executable()
    end = time.time()

    print(f"Elapsed time: {end - start}")
    print(f"Computed: {total}")
    print()
