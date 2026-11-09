"""
Description:

A NumPy array is the building block for every later week - PyTorch tensors, model weights, and datasets are 
all arrays underneath.
Create a script weeks/week01/task06.py that builds a 1D array (a vector), a 2D array (a matrix), and a 3D array
(a small volume), and prints each array's shape, dtype, ndim, and size alongside the array itself.

Acceptance criteria:

The script builds one 1D array, one 2D array, and one 3D array.
For each array, the script prints its shape, dtype, ndim, and size.
"""

import numpy as np

def main() -> int:
    one_dim_array: np.ndarray = np.array([1, 2, 3, 4, 5])
    print_characteristics(one_dim_array)

    two_dim_array: np.ndarray = np.array([[1, 2], [3, 4]])
    print_characteristics(two_dim_array)

    three_dim_array: np.ndarray = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print_characteristics(three_dim_array)
    
    return 0

def print_characteristics(array: np.ndarray) -> None:
    print(f"Array: {array.tolist()}")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    print(f"Number of dimentions: {array.ndim}")
    print(f"Size: {array.size}")
    print()