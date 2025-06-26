import torch
from openml.pytorch.losses import mse_loss

def test_mse_loss():
    pred = torch.tensor([1.0, 2.0])
    targ = torch.tensor([1.5, 2.5])
    loss = mse_loss(pred, targ)
    assert torch.isclose(loss, torch.tensor(0.25))
