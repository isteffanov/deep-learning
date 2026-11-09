"""
Description:

Create a script weeks/week01/task11.py that generates x values on a fixed range, computes y from a linear relationship with
added Gaussian noise, prints the size of the dataset and the range of x and y, then shows a scatter plot of x against y.

Acceptance criteria:

* The script generates x and y arrays of the same length, with y following a linear relationship in x plus random Gaussian noise.
* The script shows a scatter plot of x against y.
"""

import numpy as np
import matplotlib.pyplot as plt

def main() -> int:
    x: np.ndarray = np.random.randint(1, 101, 15)
    a: float = 1.8
    b: int = 22

    noise: np.ndarray = np.random.normal(0, 2, 15)

    y: np.ndarray = a*x + b + noise

    plt.scatter(x, y)
    plt.savefig("plots/week01/task11.png")

    return 0

    
