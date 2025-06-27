import numpy as np
from openml.numpy.activations import relu, sigmoid, softmax

def test_relu():
    x = np.array([-1, 0, 1])
    assert np.array_equal(relu(x), np.array([0, 0, 1]))

def test_sigmoid():
    x = np.array([-np.inf, 0, np.inf])
    out = sigmoid(x)
    assert np.allclose(out, np.array([0, 0.5, 1.0]), atol=1e-6)

def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    out = softmax(x)
    assert np.allclose(out.sum(), 1.0, atol=1e-6)
