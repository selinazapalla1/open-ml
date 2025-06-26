"""Optimizers using PyTorch."""
import torch.optim as optim

def get_sgd(parameters, lr=0.01):
    """Return an SGD optimizer."""
    return optim.SGD(parameters, lr=lr)
