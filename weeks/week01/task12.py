"""
Description:

Create a script weeks/week01/task12.py that draws two line plots (sin(x) and cos(x)) on one set of axes with a legend,
and a histogram of 1,000 samples from a normal distribution on a second set of axes, each with a title, axis labels, and a grid. Show the two side by side.

Acceptance criteria:

One figure is used to show both plots.
The required attributes are present in the plots.
"""

import numpy as np
import matplotlib.pyplot as plt

def main() -> int:
    points: np.ndarray = np.linspace(0, 10, 10_000)

    sin_x: np.ndarray = np.sin(points)
    cos_x: np.ndarray = np.cos(points)

    plt.subplot(1, 2, 1)
    plt.plot(points, sin_x)
    plt.plot(points, cos_x)
    plt.title("sin(x) and cos(x)\n from 0 to 10")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["sin(x)", "cos(x)"], loc="lower left")
    plt.grid(visible=True)


    bell: np.ndarray = np.random.normal(0, 1.6, 10_000)

    plt.subplot(1, 2, 2)
    plt.hist(bell, bins=100)
    plt.title("10 000 samples from\n a Normal Distribution")
    plt.xlabel("point")
    plt.ylabel("count")
    plt.grid(visible=True)

    plt.tight_layout()
    plt.savefig("plots/week01/task12.png")
    return 0