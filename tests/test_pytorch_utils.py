import numpy as np
import torch
from openml.pytorch.utils import to_tensor

def test_to_tensor():
    arr = np.array([1, 2, 3])
    t = to_tensor(arr)
    assert isinstance(t, torch.Tensor)
