"""Sampling utilities for text generation."""

import torch
from typing import Optional


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k sampling."""
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(top_k_probs, num_samples=1)
    next_token = torch.gather(top_k_indices, -1, next_token)
    return next_token


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample from logits with temperature, top-p, and top-k.
    
    Args:
        logits: Shape [batch_size, vocab_size]
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        
    Returns:
        Sampled token indices, shape [batch_size, 1]
    """
    if temperature == 0.0:
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Apply temperature
    probs = torch.softmax(logits / temperature, dim=-1)
    
    # Apply top-k if specified
    if top_k is not None and top_k > 0:
        return sample_top_k(probs, top_k)
    
    # Apply top-p if specified
    if top_p is not None and top_p < 1.0:
        return sample_top_p(probs, top_p)
    
    # Standard multinomial sampling
    return torch.multinomial(probs, num_samples=1)

