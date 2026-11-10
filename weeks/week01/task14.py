import numpy as np

from dl_lib.preprocessing.functional import min_max_normalize, standardize

def main() -> int:
    array: np.ndarray = np.array([2, 3, 5, 8, 13, 21])

    normalized: np.ndarray = min_max_normalize(array)
    standardized: np.ndarray = standardize(array)

    print(f"Original: {array.tolist()}")
    print(f"Normalized: {normalized.tolist()}")
    print(f"Standardized: {standardized.tolist()}")

    print()

    print(f"Normalized Min/Max: {normalized.min()}/{normalized.max()}")
    print(f"Standardized Mean/Std: {standardized.mean()}/{standardized.std()}")
    
    return 0