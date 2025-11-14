"""Example API client for nano-sglang server."""

import requests
import time


def test_sync_generation():
    """Test synchronous generation."""
    print("Testing synchronous generation...")
    
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "prompt": "The capital of France is",
            "max_tokens": 50,
            "temperature": 0.7,
        },
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated text: {result['text']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_async_generation():
    """Test asynchronous generation."""
    print("\nTesting asynchronous generation...")
    
    # Add request
    response = requests.post(
        "http://localhost:8000/generate/async",
        json={
            "prompt": "The capital of Germany is",
            "max_tokens": 50,
            "temperature": 0.7,
        },
    )
    
    if response.status_code != 200:
        print(f"Error adding request: {response.status_code} - {response.text}")
        return
    
    request_id = response.json()["request_id"]
    print(f"Request ID: {request_id}")
    
    # Poll for status
    while True:
        response = requests.get(f"http://localhost:8000/request/{request_id}")
        if response.status_code != 200:
            print(f"Error getting status: {response.status_code}")
            break
        
        status = response.json()
        print(f"Status: {status['status']}, Generated tokens: {status['generated_tokens']}")
        
        if status["status"] == "finished":
            print(f"Generated text: {status['text']}")
            break
        
        time.sleep(0.5)


def main():
    """Run all tests."""
    print("Nano-SGLang API Client Example\n")
    print("Make sure the server is running on http://localhost:8000\n")
    
    try:
        test_sync_generation()
        test_async_generation()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to server. Make sure it's running on http://localhost:8000")


if __name__ == "__main__":
    main()

