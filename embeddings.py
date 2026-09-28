import requests
import config

def get_embedding(text: str) -> list[float]:
    url = f"{config.OLLAMA_API_URL}/embeddings"
    payload = {
        "model": config.EMBEDDING_MODEL,
        "prompt": text
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]
