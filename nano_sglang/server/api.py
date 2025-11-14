"""HTTP API server for nano-sglang."""

import argparse
import asyncio
import threading
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from nano_sglang.core.engine import InferenceEngine


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    tokens: List[int]
    request_id: str


class RequestStatusResponse(BaseModel):
    request_id: str
    status: str
    generated_tokens: int
    text: str


# Global engine instance
engine: Optional[InferenceEngine] = None
engine_lock = threading.Lock()


def create_app(engine_instance: InferenceEngine) -> FastAPI:
    """Create FastAPI app with the engine."""
    global engine
    engine = engine_instance
    
    app = FastAPI(title="Nano-SGLang API", version="0.1.0")
    
    # Background task to process inference steps
    def process_inference_loop():
        """Background thread to process inference steps."""
        import time
        while True:
            if engine:
                engine.process_step()
            time.sleep(0.01)  # Small delay to avoid busy waiting
    
    # Start background thread
    inference_thread = threading.Thread(target=process_inference_loop, daemon=True)
    inference_thread.start()
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generate text from a prompt (synchronous).
        """
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        # Convert stop strings to token IDs
        stop_tokens = None
        if request.stop:
            stop_tokens = []
            for stop_str in request.stop:
                tokens = engine.model.encode(stop_str)
                stop_tokens.extend(tokens)
        
        # Generate
        text = engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_tokens=stop_tokens,
        )
        
        # Get request info (simplified - in real implementation, we'd track this better)
        return GenerateResponse(
            text=text,
            tokens=[],  # Would include actual tokens in full implementation
            request_id="sync_request",
        )
    
    @app.post("/generate/async")
    async def generate_async(request: GenerateRequest):
        """
        Add a generation request to the queue (asynchronous).
        """
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        # Convert stop strings to token IDs
        stop_tokens = None
        if request.stop:
            stop_tokens = []
            for stop_str in request.stop:
                tokens = engine.model.encode(stop_str)
                stop_tokens.extend(tokens)
        
        # Add request
        request_id = engine.add_request(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_tokens=stop_tokens,
        )
        
        return {"request_id": request_id, "status": "queued"}
    
    @app.get("/request/{request_id}", response_model=RequestStatusResponse)
    async def get_request_status(request_id: str):
        """Get status of a request."""
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        status = engine.get_request_status(request_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Request not found")
        
        return RequestStatusResponse(**status)
    
    @app.delete("/request/{request_id}")
    async def cancel_request(request_id: str):
        """Cancel a request."""
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        success = engine.scheduler.cancel_request(request_id)
        if not success:
            raise HTTPException(status_code=404, detail="Request not found")
        
        return {"status": "cancelled", "request_id": request_id}
    
    return app


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="Nano-SGLang API Server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Disable RadixAttention cache",
    )
    
    args = parser.parse_args()
    
    # Initialize engine
    print("Initializing inference engine...")
    engine = InferenceEngine(
        model_path=args.model_path,
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        enable_radix_cache=not args.disable_radix_cache,
    )
    
    # Create app
    app = create_app(engine)
    
    # Run server
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

