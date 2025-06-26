"""Optimizers using NumPy."""
import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        """Update parameters in-place."""
        for p in self.params:
            p -= self.lr * p.grad
