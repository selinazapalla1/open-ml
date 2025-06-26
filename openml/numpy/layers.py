"""Layer definitions using NumPy."""
import numpy as np

class Dense:
    """A fully connected neural network layer."""
    def __init__(self, input_dim, output_dim):
        # Initialize weights and biases
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, x):
        """Forward pass for dense layer."""
        return x @ self.weights + self.bias
