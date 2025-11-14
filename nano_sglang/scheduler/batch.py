"""Batch scheduler for continuous batching."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time


class RequestStatus(Enum):
    """Status of a request."""
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class Request:
    """A single generation request."""
    
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    max_tokens: int
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_tokens: List[int] = field(default_factory=list)
    
    # Generation state
    generated_token_ids: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    created_at: float = field(default_factory=time.time)
    
    # KV cache info
    kv_cache_start_pos: int = 0
    kv_cache_end_pos: int = 0
    prefix_match_length: int = 0  # Length of matched prefix from RadixCache
    
    def is_finished(self) -> bool:
        """Check if request is finished."""
        if self.status == RequestStatus.FINISHED:
            return True
        if self.status == RequestStatus.CANCELLED:
            return True
        if len(self.generated_token_ids) >= self.max_tokens:
            return True
        if self.generated_token_ids and self.generated_token_ids[-1] in self.stop_tokens:
            return True
        return False
    
    def get_full_token_ids(self) -> List[int]:
        """Get full token sequence (prompt + generated)."""
        return self.prompt_token_ids + self.generated_token_ids


@dataclass
class Batch:
    """A batch of requests for processing."""
    
    requests: List[Request]
    input_ids: List[List[int]]  # Token IDs for each request
    attention_mask: List[List[int]]  # Attention mask for each request
    seq_lens: List[int]  # Sequence length for each request
    max_seq_len: int  # Maximum sequence length in batch
    
    def __len__(self) -> int:
        return len(self.requests)
    
    def is_empty(self) -> bool:
        return len(self.requests) == 0


class BatchScheduler:
    """
    Simple batch scheduler for continuous batching.
    
    This scheduler manages requests and creates batches for inference.
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        max_waiting_time: float = 0.1,  # Max time to wait before batching
    ):
        """
        Initialize the batch scheduler.
        
        Args:
            max_batch_size: Maximum number of requests in a batch
            max_seq_len: Maximum sequence length
            max_waiting_time: Maximum time to wait before creating a batch
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_waiting_time = max_waiting_time
        
        self.waiting_queue: List[Request] = []
        self.running_requests: Dict[str, Request] = {}
        self.finished_requests: Dict[str, Request] = {}
        
        self.request_counter = 0
    
    def add_request(
        self,
        prompt: str,
        prompt_token_ids: List[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_tokens: Optional[List[int]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Add a new request to the queue.
        
        Returns:
            Request ID
        """
        if request_id is None:
            self.request_counter += 1
            request_id = f"req_{self.request_counter}"
        
        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_tokens=stop_tokens or [],
        )
        
        self.waiting_queue.append(request)
        return request_id
    
    def get_next_batch(self) -> Optional[Batch]:
        """
        Get the next batch to process.
        
        Returns:
            Batch object or None if no requests available
        """
        if not self.waiting_queue:
            return None
        
        # Select requests for batching
        selected_requests = []
        current_time = time.time()
        
        for req in self.waiting_queue:
            # Check if we should wait for more requests
            if len(selected_requests) >= self.max_batch_size:
                break
            
            # Check if request is too long
            if len(req.prompt_token_ids) > self.max_seq_len:
                req.status = RequestStatus.CANCELLED
                self.finished_requests[req.request_id] = req
                continue
            
            # Add request if it's been waiting long enough or batch is getting full
            wait_time = current_time - req.created_at
            if wait_time >= self.max_waiting_time or len(selected_requests) == 0:
                selected_requests.append(req)
        
        if not selected_requests:
            return None
        
        # Remove selected requests from waiting queue
        for req in selected_requests:
            self.waiting_queue.remove(req)
            req.status = RequestStatus.RUNNING
            self.running_requests[req.request_id] = req
        
        # Create batch
        return self._create_batch(selected_requests)
    
    def _create_batch(self, requests: List[Request]) -> Batch:
        """Create a batch from a list of requests."""
        input_ids = []
        attention_mask = []
        seq_lens = []
        
        for req in requests:
            # Get full token sequence (prompt + generated so far)
            full_tokens = req.get_full_token_ids()
            
            # Pad to max length in batch
            seq_len = len(full_tokens)
            seq_lens.append(seq_len)
            
            # For simplicity, we'll pad to the max sequence length in this batch
            # In a real implementation, we'd use more sophisticated padding
            max_len = max(seq_lens)
            max_len = min(max_len, self.max_seq_len)
            
            padded_tokens = full_tokens[:max_len] + [0] * (max_len - len(full_tokens))
            mask = [1] * seq_len + [0] * (max_len - seq_len)
            
            input_ids.append(padded_tokens)
            attention_mask.append(mask)
        
        max_seq_len = max(seq_lens)
        
        return Batch(
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
    
    def update_request(
        self,
        request_id: str,
        new_token_id: int,
        finished: bool = False,
    ):
        """Update a request with a new generated token."""
        if request_id not in self.running_requests:
            return
        
        request = self.running_requests[request_id]
        request.generated_token_ids.append(new_token_id)
        
        if finished or request.is_finished():
            request.status = RequestStatus.FINISHED
            self.finished_requests[request_id] = request
            del self.running_requests[request_id]
    
    def get_request(self, request_id: str) -> Optional[Request]:
        """Get a request by ID."""
        if request_id in self.running_requests:
            return self.running_requests[request_id]
        if request_id in self.finished_requests:
            return self.finished_requests[request_id]
        for req in self.waiting_queue:
            if req.request_id == request_id:
                return req
        return None
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request."""
        request = self.get_request(request_id)
        if request is None:
            return False
        
        request.status = RequestStatus.CANCELLED
        if request_id in self.waiting_queue:
            self.waiting_queue.remove(request)
        if request_id in self.running_requests:
            del self.running_requests[request_id]
        self.finished_requests[request_id] = request
        return True

