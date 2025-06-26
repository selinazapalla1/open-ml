import torch
from openml.pytorch.optimizers import get_sgd

def test_get_sgd():
    model = torch.nn.Linear(1, 1)
    opt = get_sgd(model.parameters(), lr=0.1)
    assert isinstance(opt, torch.optim.SGD)
