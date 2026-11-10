"""
Description:

Create a script weeks/week01/task15.py that builds a NumPy array, converts it to a PyTorch tensor, prints both arrays' dtype and
shape alongside the tensor's device, then selects the GPU device if available (mps on this machine, otherwise cpu) and moves the
tensor there, printing the tensor's device and dtype after the move.

Acceptance criteria:

* The script converts a NumPy array into a PyTorch tensor and prints both arrays' dtype and shape.
* The script prints the tensor's device before and after selecting and moving to the available GPU device (or cpu if none is available).
"""

import numpy as np
import torch

def main() -> int:
    array: np.ndarray = np.random.normal(size=1024)

    print(f"Array's dtype: {array.dtype}")
    print(f"Array's shape: {array.shape}")
    print()

    tensor: torch.Tensor = torch.as_tensor(array)

    print(f"Tensor's dtype: {tensor.dtype}")
    print(f"Tensor's shape: {tensor.shape}")
    print(f"Device, containing the tensor: {tensor.device}")
    print()

    mps: torch.Device = torch.device("mps")
    tensor_on_mps: torch.Tensor = tensor.to(mps, dtype=torch.float32) # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.

    print(f"Tensor on mps's dtype: {tensor_on_mps.dtype}")
    print(f"Tensor on mps's shape: {tensor_on_mps.shape}")
    print(f"Device, containing the tensor: {tensor_on_mps.device}")

    return 0
