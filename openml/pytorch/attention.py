"""Scaled dot-product attention using PyTorch."""
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v):
    """
    Compute scaled dot-product attention.
    q, k, v: tensors of shape (..., seq_len, depth)
    returns: output, attention_weights
    """
    # Compute raw scores
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    # Scale scores
    scaled_scores = matmul_qk / (dk ** 0.5)
    # Softmax to get weights
    weights = F.softmax(scaled_scores, dim=-1)
    # Weighted sum of values
    output = torch.matmul(weights, v)
    return output, weights
