"""Layer definitions using PyTorch."""
import torch.nn as nn

class Dense(nn.Module):
    """A fully connected neural network layer."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """Forward pass for dense layer."""
        return self.linear(x)
