"""
API Server example for AI Assistant Pro

Demonstrates:
- Starting the FastAPI server
- Making API requests
- Streaming responses
"""

from ai_assistant_pro.serving import serve
import requests
import json


def start_server():
    """
    Start the API server

    Run this in a separate terminal:
    $ python examples/api_server.py
    """
    serve(
        host="0.0.0.0",
        port=8000,
        model_name="gpt2",
        use_triton=True,
        use_fp8=False,
    )


def example_client():
    """
    Example client for making requests to the API

    Run this after starting the server
    """
    base_url = "http://localhost:8000"

    print("AI Assistant Pro - API Client Example")
    print("=" * 60)

    # 1. Health check
    print("\n1. Health check:")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    # 2. Generate text
    print("\n2. Generate text:")
    request_data = {
        "prompt": "The future of AI is",
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.9,
    }

    response = requests.post(f"{base_url}/generate", json=request_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Generated text: {result['text']}")
    print(f"Tokens: {result['tokens']}")

    # 3. OpenAI-compatible endpoint
    print("\n3. OpenAI-compatible endpoint:")
    response = requests.post(f"{base_url}/v1/completions", json=request_data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    # 4. Get statistics
    print("\n4. Engine statistics:")
    response = requests.get(f"{base_url}/stats")
    print(json.dumps(response.json(), indent=2))

    print("\n✓ API client example complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "client":
        example_client()
    else:
        print("Starting API server...")
        print("Run 'python examples/api_server.py client' in another terminal to test the API")
        start_server()
