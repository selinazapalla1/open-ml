"""Activation functions using PyTorch."""
import torch
import torch.nn.functional as F

def relu(x):
    """Rectified Linear Unit."""
    return F.relu(x)

def sigmoid(x):
    """Sigmoid activation function."""
    return torch.sigmoid(x)

def softmax(x):
    """Softmax activation along the last dimension."""
    return F.softmax(x, dim=-1)
