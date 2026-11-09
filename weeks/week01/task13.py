"""
Description:

Create a script weeks/week01/task13.py that:

Draws 10,000 samples each from a uniform, a normal, and an exponential distribution.
Prints the mean, standard deviation, min, and max of each.
Shows a histogram of the three side by side.

Acceptance criteria:

The script draws samples from a uniform, a normal, and an exponential distribution.
"""

import numpy as np
import matplotlib.pyplot as plt

def main() -> int:
    sample_size: int = 10_000

    uniform_samples: np.ndarray = np.random.uniform(size=sample_size)
    normal_samples: np.ndarray = np.random.normal(size=sample_size)
    exponential_samples: np.ndarray = np.random.exponential(size=sample_size)

    describe_distribution(uniform_samples, "Uniform Distribution")
    describe_distribution(normal_samples, "Normal Distribution")
    describe_distribution(exponential_samples, "Exponential Distribution")

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharex=False, sharey=False)
    axs[0].hist(uniform_samples, bins=100)
    axs[0].set_title("Uniform Distribution")
    axs[0].grid()

    axs[1].hist(normal_samples, bins=100)
    axs[1].set_title("Normal Distribution")
    axs[1].grid()

    axs[2].hist(exponential_samples, bins=100)
    axs[2].set_title("Exponential Distribution")
    axs[2].grid()
    
    fig.supxlabel('Value')
    fig.supylabel('Count')

    fig.tight_layout()
    fig.savefig("plots/week01/task13.png")

    return 0

def describe_distribution(dist: np.ndarray, name: str) -> None:
    print(name)
    print(f"Mean: {dist.mean()}")
    print(f"Standard Deviation: {dist.std()}")
    print(f"Min: {dist.min()}")
    print(f"Max: {dist.max()}")
    print()

