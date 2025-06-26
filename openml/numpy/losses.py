"""Loss functions using NumPy."""
import numpy as np

def mse_loss(predictions, targets):
    """Mean Squared Error loss."""
    return np.mean((predictions - targets) ** 2)
