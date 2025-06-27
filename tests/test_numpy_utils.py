import numpy as np
import torch
from openml.numpy.utils import to_numpy

def test_to_numpy():
    t = torch.tensor([1, 2, 3])
    arr = to_numpy(t)
    assert isinstance(arr, np.ndarray)
