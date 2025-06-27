import numpy as np
from openml.numpy.losses import mse_loss

def test_mse_loss():
    pred = np.array([1.0, 2.0])
    targ = np.array([1.5, 2.5])
    loss = mse_loss(pred, targ)
    assert np.isclose(loss, 0.25)
