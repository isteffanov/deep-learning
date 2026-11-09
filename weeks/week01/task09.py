"""
Description:

Create a script weeks/week01/task09.py that adds a scalar, a row vector, and a column vector to the same matrix,
printing each pair of shapes and the result. It then attempts to add a vector whose shape cannot be broadcast
against the matrix and prints the resulting error.

Acceptance criteria:

The script adds a scalar to a matrix and prints the result.
The script adds a row vector and a column vector to the same matrix, printing each operand's shape and the result.
The script attempts to add an incompatible shape and prints the ValueError it raises.
"""

import numpy as np


def main() -> int:
    matrix: np.ndarray = np.array([[1, 2, 3],
                                   [4, 5, 6]])

    scalar: int = 10
    print(f"Matrix + scalar ({scalar}):")
    print(matrix + scalar)
    print()

    row_vector: np.ndarray = np.array([10, 20, 30])
    print_sum(matrix, row_vector, "row vector")

    column_vector: np.ndarray = np.array([[100],
                                          [200]])
    print_sum(matrix, column_vector, "column vector")

    incompatible: np.ndarray = np.array([1, 2])
    print(f"Matrix {matrix.shape} + incompatible {incompatible.shape} raises:")
    try:
        matrix + incompatible
    except ValueError as error:
        print(error)

    return 0


def print_sum(matrix: np.ndarray, vector: np.ndarray, name: str) -> None:
    print(f"Matrix {matrix.shape} + {name} {vector.shape}:")
    print(matrix + vector)
    print()
