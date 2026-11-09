"""
Description:

Create a script weeks/week01/task10.py that builds two small square matrices and prints their elementwise product,
their matrix product, and the transpose of the first matrix.

Acceptance criteria:

The script prints the elementwise product of two matrices.
The script prints the matrix product of the same two matrices.
The script prints the transpose of one of the matrices.
"""

import numpy as np


def main() -> int:
    a: np.ndarray = np.array([[1, 2],
                              [3, 4]])
    b: np.ndarray = np.array([[5, 6],
                              [7, 8]])

    print_matrix("A", a)
    print_matrix("B", b)

    print_matrix("Elementwise", a * b)

    print_matrix("Matrix multiplication", a @ b)

    print_matrix("The transpose of A", a.T, end="")

    return 0


def print_matrix(title: str, matrix: np.ndarray, end: str = "\n") -> None:
    print(f"{title}:")
    print(matrix)
    print(end=end)
