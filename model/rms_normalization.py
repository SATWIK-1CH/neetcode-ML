import numpy as np
from typing import List


class Solution:
    def rms_norm(
        self,
        x: List[float],
        gamma: List[float],
        eps: float = 1e-5
    ) -> List[float]:
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        n = len(x)

        # RMS = sqrt(mean(x^2) + eps)
        rms = np.sqrt(np.sum(x ** 2) / n + eps)

        # Normalize each value
        x_hat = x / rms

        # Elementwise scale by gamma
        output = gamma * x_hat

        # Round and convert back to a Python list
        return np.round(output, 4).tolist()