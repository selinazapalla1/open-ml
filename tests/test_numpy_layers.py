import numpy as np
from openml.numpy.layers import Dense

def test_dense_forward():
    np.random.seed(0)
    layer = Dense(3, 2)
    x = np.ones((1, 3))
    out = layer.forward(x)
    assert out.shape == (1, 2)
