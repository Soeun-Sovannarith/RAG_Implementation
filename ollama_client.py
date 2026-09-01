import json
import requests
from typing import Generator, List, Dict, Any

OLLAMA_BASE_URL = "http://localhost:11434"

def get_embedding(text: str, model: str = "mxbai-embed-large") -> List[float]:
    """
    Fetches a dense vector embedding for a given text string directly from Ollama.
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])
    except Exception as e:
        print(f"Error fetching embedding from Ollama ({model}): {e}")
        return []

def stream_chat(messages: List[Dict[str, str]], model: str = "llama3.2") -> Generator[str, None, None]:
    """
    Streams chat completion tokens from Ollama's /api/chat endpoint.
    messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    msg = chunk.get("message", {})
                    token = msg.get("content", "")
                    yield token
                    if chunk.get("done", False):
                        break
    except Exception as e:
        yield f"\n[Error connecting to Ollama: {e}]"

def generate_chat(messages: List[Dict[str, str]], model: str = "llama3.2") -> str:
    """
    Synchronously fetches full response from Ollama.
    """
    full_response = ""
    for token in stream_chat(messages, model=model):
        full_response += token
    return full_response

if __name__ == "__main__":
    print("Testing Ollama API Client...")
    emb = get_embedding("Hello world")
    print(f"Embedding dimensions: {len(emb)}")
    
    print("\nTesting Streaming Chat:")
    for token in stream_chat([{"role": "user", "content": "Say hello in 3 words"}]):
        print(token, end="", flush=True)
    print()
