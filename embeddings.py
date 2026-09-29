import requests

import config


def get_embedding(text: str) -> list[float]:
    """Text -> vector via Ollama."""
    response = requests.post(
        f"{config.OLLAMA_API_URL}/embeddings",
        json={"model": config.EMBEDDING_MODEL, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embedding"]