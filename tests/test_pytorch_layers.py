import torch
from openml.pytorch.layers import Dense

def test_dense_forward():
    torch.manual_seed(0)
    layer = Dense(3, 2)
    x = torch.ones(1, 3)
    out = layer(x)
    assert out.shape == (1, 2)
