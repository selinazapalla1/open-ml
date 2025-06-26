"""Activation functions using NumPy."""
import numpy as np

def relu(x):
    """Rectified Linear Unit."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """Softmax activation along the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
