"""Loss functions using PyTorch."""
import torch.nn.functional as F

def mse_loss(predictions, targets):
    """Mean Squared Error loss."""
    return F.mse_loss(predictions, targets)
