"""Model wrapper for LLM inference."""

import torch
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class ModelWrapper:
    """Wrapper for HuggingFace models with KV cache support."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 2048,
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the model or HuggingFace model ID
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type for model weights
            max_seq_len: Maximum sequence length
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Model info
        self.num_layers = getattr(self.config, 'num_hidden_layers', getattr(self.config, 'num_layers', 32))
        self.num_heads = getattr(self.config, 'num_attention_heads', 32)
        self.head_dim = getattr(self.config, 'hidden_size', 4096) // self.num_heads
        self.vocab_size = self.config.vocab_size
        self.hidden_size = getattr(self.config, 'hidden_size', 4096)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            past_key_values: Cached key-value pairs from previous forward passes
            use_cache: Whether to return key-value cache
            
        Returns:
            Tuple of (logits, past_key_values)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        
        return outputs.logits, outputs.past_key_values
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

