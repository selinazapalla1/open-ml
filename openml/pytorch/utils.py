"""Utility functions using PyTorch."""
import torch

def to_tensor(array, device=None):
    """Convert NumPy array to PyTorch tensor."""
    return torch.tensor(array, device=device)
