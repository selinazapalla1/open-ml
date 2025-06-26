"""Scaled dot-product attention using NumPy."""
import numpy as np

def scaled_dot_product_attention(q, k, v):
    """
    Compute scaled dot-product attention.
    q, k, v: numpy arrays of shape (..., seq_len, depth)
    returns: output, attention_weights
    """
    # Compute raw scores
    matmul_qk = np.matmul(q, np.swapaxes(k, -2, -1))
    dk = k.shape[-1]
    # Scale scores
    scaled_scores = matmul_qk / np.sqrt(dk)
    # Apply softmax to get weights
    weights = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    # Weighted sum of values
    output = np.matmul(weights, v)
    return output, weights
