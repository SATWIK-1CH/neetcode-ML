
import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert inputs to NumPy arrays
        x = np.array(x, dtype=float)
        w1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        w2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # -------------------------
        # FORWARD PROPAGATION
        # -------------------------

        # First linear layer
        z1 = x @ w1.T + b1

        # ReLU activation
        a1 = np.maximum(0, z1)

        # Second linear layer
        z2 = a1 @ w2.T + b2

        # Mean Squared Error
        loss = np.mean((z2 - y_true) ** 2)

        # -------------------------
        # BACKPROPAGATION
        # -------------------------

        # Gradient of MSE
        n = len(y_true)
        dz2 = 2 * (z2 - y_true) / n

        # Gradients for second layer
        dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1)
        db2 = dz2

        # Backpropagate to hidden layer
        da1 = w2.T @ dz2

        # Derivative of ReLU
        mask = (z1 > 0).astype(float)
        dz1 = da1 * mask

        # Gradients for first layer
        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
        db1 = dz1

        # Return rounded results
        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }
