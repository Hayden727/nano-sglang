"""Core inference engine for nano-sglang."""

import torch
from typing import List, Optional, Dict, Any
import time

from nano_sglang.core.model import ModelWrapper
from nano_sglang.scheduler.batch import BatchScheduler, Request, Batch
from nano_sglang.cache.radix import RadixCache, RadixKey
from nano_sglang.utils.sampling import sample
from nano_sglang.utils.kv_cache import KVCache


class InferenceEngine:
    """
    Core inference engine with RadixAttention and continuous batching.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        enable_radix_cache: bool = True,
        radix_cache_size: int = 10000,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model or HuggingFace model ID
            device: Device to run on
            dtype: Data type for model weights
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            enable_radix_cache: Enable RadixAttention prefix caching
            radix_cache_size: Size of radix cache
        """
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = ModelWrapper(model_path, device, dtype, max_seq_len)
        print("Model loaded successfully!")
        
        # Initialize scheduler
        self.scheduler = BatchScheduler(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        
        # Initialize RadixAttention cache
        self.radix_cache: Optional[RadixCache] = None
        if enable_radix_cache:
            self.radix_cache = RadixCache(
                max_size=radix_cache_size,
                eviction_policy="lru",
            )
            print("RadixAttention cache enabled!")
        
        # KV cache storage (simplified - in real implementation, this would be more sophisticated)
        self.kv_cache_storage: Dict[str, Any] = {}
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt (synchronous, blocking).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_tokens: List of stop token IDs
            request_id: Optional request ID
            
        Returns:
            Generated text
        """
        # Encode prompt
        prompt_token_ids = self.model.encode(prompt)
        
        # Add request to scheduler
        req_id = self.scheduler.add_request(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_tokens=stop_tokens or [self.model.tokenizer.eos_token_id],
            request_id=request_id,
        )
        
        # Process until finished
        generated_tokens = []
        while True:
            # Get next batch
            batch = self.scheduler.get_next_batch()
            if batch is None:
                # No requests in queue, check if our request is finished
                request = self.scheduler.get_request(req_id)
                if request and request.is_finished():
                    break
                time.sleep(0.01)  # Small sleep to avoid busy waiting
                continue
            
            # Process batch
            results = self._process_batch(batch)
            
            # Update requests
            for req, token_id in zip(batch.requests, results):
                if req.request_id == req_id:
                    generated_tokens.append(token_id)
                self.scheduler.update_request(
                    req.request_id,
                    token_id,
                    finished=req.is_finished(),
                )
            
            # Check if our request is finished
            request = self.scheduler.get_request(req_id)
            if request and request.is_finished():
                break
        
        # Decode generated tokens
        return self.model.decode(generated_tokens)
    
    def _process_batch(self, batch: Batch) -> List[int]:
        """
        Process a batch of requests.
        
        Args:
            batch: Batch to process
            
        Returns:
            List of next token IDs for each request
        """
        # Convert to tensors
        input_ids_tensor = torch.tensor(
            batch.input_ids,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask_tensor = torch.tensor(
            batch.attention_mask,
            dtype=torch.long,
            device=self.device,
        )
        
        # Check for prefix matches in RadixCache
        past_key_values_list = []
        for req in batch.requests:
            if self.radix_cache:
                # Try to match prefix
                full_tokens = req.get_full_token_ids()
                key = RadixKey(token_ids=full_tokens)
                matched_indices, last_node = self.radix_cache.match_prefix(key)
                
                if matched_indices:
                    # We have a prefix match - in a real implementation,
                    # we would load the cached KV values here
                    req.prefix_match_length = len(matched_indices)
                    # For now, we'll just skip the cached tokens
                    # In a full implementation, we'd use the cached KV cache
                    past_key_values = None
                else:
                    past_key_values = None
            else:
                past_key_values = None
            
            past_key_values_list.append(past_key_values)
        
        # For simplicity, we'll process all requests without prefix caching
        # In a real implementation, we'd handle prefix caching more carefully
        past_key_values = None
        
        # Forward pass
        logits, new_past_key_values = self.model.forward(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        # Get next token logits (last token of each sequence)
        next_token_logits = []
        for i, seq_len in enumerate(batch.seq_lens):
            next_token_logits.append(logits[i, seq_len - 1, :])
        
        next_token_logits = torch.stack(next_token_logits)
        
        # Sample next tokens
        next_token_ids = []
        for i, req in enumerate(batch.requests):
            token_id = sample(
                next_token_logits[i:i+1],
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
            ).item()
            next_token_ids.append(token_id)
        
        # Update RadixCache
        if self.radix_cache:
            for req in batch.requests:
                full_tokens = req.get_full_token_ids()
                if len(full_tokens) > req.prefix_match_length:
                    # Store new prefix in cache
                    # In a real implementation, we'd store the actual KV cache indices
                    new_tokens = full_tokens[req.prefix_match_length:]
                    key = RadixKey(token_ids=new_tokens)
                    # For now, we'll just use dummy indices
                    kv_indices = list(range(len(new_tokens)))
                    self.radix_cache.insert(key, kv_indices, last_node)
        
        return next_token_ids
    
    def add_request(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Add a request to the queue (non-blocking).
        
        Returns:
            Request ID
        """
        prompt_token_ids = self.model.encode(prompt)
        return self.scheduler.add_request(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_tokens=stop_tokens or [self.model.tokenizer.eos_token_id],
            request_id=request_id,
        )
    
    def process_step(self) -> Dict[str, Any]:
        """
        Process one step of inference (non-blocking).
        
        Returns:
            Dictionary with results
        """
        batch = self.scheduler.get_next_batch()
        if batch is None:
            return {"status": "idle"}
        
        results = self._process_batch(batch)
        
        # Update requests
        finished_requests = []
        for req, token_id in zip(batch.requests, results):
            self.scheduler.update_request(
                req.request_id,
                token_id,
                finished=req.is_finished(),
            )
            if req.is_finished():
                finished_requests.append({
                    "request_id": req.request_id,
                    "text": self.model.decode(req.generated_token_ids),
                    "tokens": req.generated_token_ids,
                })
        
        return {
            "status": "processed",
            "batch_size": len(batch),
            "finished_requests": finished_requests,
        }
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a request."""
        request = self.scheduler.get_request(request_id)
        if request is None:
            return None
        
        return {
            "request_id": request.request_id,
            "status": request.status.value,
            "generated_tokens": len(request.generated_token_ids),
            "text": self.model.decode(request.generated_token_ids) if request.generated_token_ids else "",
        }

