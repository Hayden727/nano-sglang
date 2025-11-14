"""KV Cache management utilities."""

import torch
from typing import List, Tuple, Optional


class KVCache:
    """Simple KV cache for storing key-value pairs."""
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        
        # Initialize cache: [num_layers, 2, max_seq_len, num_heads, head_dim]
        # 2 for key and value
        self.cache = torch.zeros(
            (num_layers, 2, max_seq_len, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.current_length = 0
    
    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        start_pos: int,
    ):
        """
        Update KV cache for a specific layer.
        
        Args:
            layer_idx: Layer index
            new_k: New key tensor, shape [batch_size, num_heads, seq_len, head_dim]
            new_v: New value tensor, shape [batch_size, num_heads, seq_len, head_dim]
            start_pos: Starting position in cache
        """
        seq_len = new_k.shape[2]
        end_pos = start_pos + seq_len
        
        # Store keys and values
        # new_k/v: [batch_size, num_heads, seq_len, head_dim]
        # cache: [num_layers, 2, max_seq_len, num_heads, head_dim]
        # For simplicity, we assume batch_size=1 in decode phase
        if new_k.shape[0] == 1:
            self.cache[layer_idx, 0, start_pos:end_pos] = new_k[0].transpose(0, 1)
            self.cache[layer_idx, 1, start_pos:end_pos] = new_v[0].transpose(0, 1)
        else:
            # For prefill, we need to handle multiple sequences
            # This is a simplified version
            for i in range(new_k.shape[0]):
                self.cache[layer_idx, 0, start_pos:end_pos] = new_k[i].transpose(0, 1)
                self.cache[layer_idx, 1, start_pos:end_pos] = new_v[i].transpose(0, 1)
    
    def get(
        self,
        layer_idx: int,
        start_pos: int,
        end_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached keys and values.
        
        Args:
            layer_idx: Layer index
            start_pos: Start position
            end_pos: End position
            
        Returns:
            Tuple of (keys, values), each shape [num_heads, seq_len, head_dim]
        """
        k = self.cache[layer_idx, 0, start_pos:end_pos].transpose(0, 1)
        v = self.cache[layer_idx, 1, start_pos:end_pos].transpose(0, 1)
        return k, v
    
    def clear(self):
        """Clear the cache."""
        self.cache.zero_()
        self.current_length = 0

