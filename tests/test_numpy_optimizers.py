import numpy as np
from openml.numpy.optimizers import SGD

def test_sgd_step():
    class Param:
        def __init__(self):
            self.data = np.array([1.0])
            self.grad = np.array([0.1])
    p = Param()
    optimizer = SGD([p], lr=0.1)
    optimizer.step()
    assert np.isclose(p.data, 1.0 - 0.1 * 0.1)
