import torch
from openml.pytorch.activations import relu, sigmoid, softmax

def test_relu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = relu(x)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0]))

def test_sigmoid():
    x = torch.tensor([-1000.0, 0.0, 1000.0])
    out = sigmoid(x)
    assert torch.allclose(out[0], torch.tensor(0.0), atol=1e-6)

def test_softmax_sum():
    x = torch.tensor([1.0, 2.0, 3.0])
    out = softmax(x)
    assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-6)
