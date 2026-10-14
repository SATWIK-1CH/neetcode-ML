
import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        # Avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate binary cross-entropy
        loss = -np.mean(
            y_true * np.log(y_pred_clipped)
            + (1 - y_true) * np.log(1 - y_pred_clipped)
        )

        return round(float(loss), 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        # Avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate loss for each sample
        sample_losses = -np.sum(
            y_true * np.log(y_pred_clipped),
            axis=1
        )

        # Average loss across samples
        total_loss = np.mean(sample_losses)

        return round(float(total_loss), 4)
