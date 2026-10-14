import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sub=NDArray[np.float64]
        sub=z-np.max(z)
        sub_sum=np.sum(np.exp(sub))
        prob=np.exp(sub)/sub_sum
        return np.round(prob,4)
