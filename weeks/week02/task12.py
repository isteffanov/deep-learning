"""
Create a script weeks/week2/task12.py that computes softmax on a small set of 
scores and confirms the probabilities sum to 1, then computes a naive softmax
(exponentiate first, no subtraction) on large scores and shows it producing nan,
next to the stable version producing the identical, finite result.

Show how torch.softmax can be used to obtain the same results.

Acceptance criteria:

* Calculations are done via numpy.
* PyTorch is used to show the built-in way of calculating the same functions.
* The script shows a naive softmax producing nan on large scores, next to softmax 
	producing the same finite probabilities it would on the unshifted scores.

"""

import torch
import numpy as np

from dl_lib.activations.functional import softmax, unsafe_softmax

def main() -> int:

	scores: np.ndarray = np.array([10, 1000, 4, 50], dtype=np.float32)
	probs: np.ndarray = softmax(scores)

	print(f"scores:	{scores}")
	print(f"probs:	{probs}")
	print(f"sum of probs:	{np.sum(probs)}")

	print()
	print(f"unsafe softmax: {unsafe_softmax(scores)}")

	torch.nn.functional.softmax(torch.from_numpy(scores), dim=0)

	return 0