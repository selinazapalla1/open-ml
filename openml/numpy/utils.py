"""Utility functions using NumPy."""
import numpy as np

def to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array if needed."""
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return tensor
