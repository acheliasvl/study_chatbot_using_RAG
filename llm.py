import json

import requests

import config


def check_ollama() -> list[str]:
    """Return a list of problems (empty list = all good)."""
    try:
        r = requests.get(f"{config.OLLAMA_API_URL}/tags", timeout=3)
        r.raise_for_status()
    except requests.RequestException:
        return ["Ollama not reachable at " + config.OLLAMA_API_URL + "  (start it: `ollama serve`)"]

    installed = {m["name"] for m in r.json().get("models", [])}
    problems = []
    for model in (config.LLM_MODEL, config.EMBEDDING_MODEL):
        if model not in installed and f"{model}:latest" not in installed:
            problems.append(f"Model missing: {model}  (run: ollama pull {model})")
    return problems


def generate_hypothetical_questions(chunk_text: str, n: int = 3) -> str:
    """Reverse-HyDE: questions this chunk could answer. Embedded together with the chunk."""
    prompt = (
        f"Generate {n} hypothetical questions that this text can answer. "
        f"Output only the questions, one per line:\n\n{chunk_text}"
    )
    response = requests.post(
        f"{config.OLLAMA_API_URL}/generate",
        json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_answer_stream(prompt: str):
    """Yield the answer token by token."""
    response = requests.post(
        f"{config.OLLAMA_API_URL}/generate",
        json={
            "model": config.LLM_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.2},  # low = stick to the context
        },
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if line:
            yield json.loads(line).get("response", "")