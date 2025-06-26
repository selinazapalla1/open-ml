import numpy as np
from openml.dumpy.attention import scaled_dot_product_attention

def test_attention_shapes():
    q = k = v = np.random.rand(2, 4, 8)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 8)
    assert weights.shape == (2, 4, 4)
