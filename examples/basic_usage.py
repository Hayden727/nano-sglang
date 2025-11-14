"""Basic usage example for nano-sglang."""

from nano_sglang.core.engine import InferenceEngine


def main():
    # Initialize engine
    print("Initializing engine...")
    engine = InferenceEngine(
        model_path="meta-llama/Llama-2-7b-chat-hf",  # Replace with your model path
        device="cuda",
        max_batch_size=8,
        max_seq_len=2048,
        enable_radix_cache=True,
    )
    
    # Generate text
    print("\nGenerating text...")
    result = engine.generate(
        prompt="The capital of France is",
        max_tokens=50,
        temperature=0.7,
    )
    
    print(f"\nGenerated text: {result}")
    
    # Test with multiple requests
    print("\n\nTesting with multiple requests...")
    request_ids = []
    
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
    ]
    
    for prompt in prompts:
        req_id = engine.add_request(
            prompt=prompt,
            max_tokens=30,
            temperature=0.7,
        )
        request_ids.append(req_id)
        print(f"Added request {req_id} for prompt: {prompt}")
    
    # Process requests
    print("\nProcessing requests...")
    while request_ids:
        result = engine.process_step()
        if result["status"] == "processed":
            for finished_req in result.get("finished_requests", []):
                if finished_req["request_id"] in request_ids:
                    print(f"\nRequest {finished_req['request_id']} finished:")
                    print(f"  Text: {finished_req['text']}")
                    request_ids.remove(finished_req["request_id"])
    
    print("\nAll requests completed!")


if __name__ == "__main__":
    main()

