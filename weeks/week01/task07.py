"""
Description:

Create a script weeks/week01/task07.py that builds a small 2D matrix and prints:

* one row;
* one column;
* a rectangular submatrix (a slice of rows and a slice of columns together);
* a boolean mask built from a comparison on the matrix, and the values the mask selects.

Acceptance criteria:

The script builds a 2D matrix and prints a single row and a single column extracted from it.
The script prints a submatrix extracted with a row slice and a column slice together.
The script builds a boolean mask from a comparison on the matrix and prints both the mask and the values it selects.
"""

import numpy as np

def main() -> int:
    array: np.ndarray = np.array([[1, 2, 3, 4, 5],
                                  [6, 7, 8, 9, 0]])
    print(f"Matrix: {array.tolist()}")

    print(f"Second row: {array[1]}")
    print(f"Third column: {array.T[2]}")
    print(f"Sub-matrix: {array[:, 1:3]}")
    print(f"Even numbers mask: {array % 2 == 0}")
    return 0