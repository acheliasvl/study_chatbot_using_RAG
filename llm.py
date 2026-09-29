import json

import requests

import config

import re

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


def generate_hypothetical_questions(
    chunk_text: str,
    chunk_id: int,
    n: int = 3
) -> list[dict]:
    """Generate individual Reverse-HyDE questions for a chunk."""

    prompt = (
        f"Generate {n} different hypothetical questions that this text can answer.\n"
        f"Each question should resemble a realistic question a student might ask.\n"
        f"Output only the questions, one per line.\n\n"
        f"{chunk_text}"
    )

    response = requests.post(
        f"{config.OLLAMA_API_URL}/generate",
        json={
            "model": config.LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=300,
    )

    response.raise_for_status()

    raw = response.json().get("response", "").strip()

    # Convert model output into individual questions
    questions = [
        line.strip()
        for line in raw.splitlines()
        if line.strip()
    ]

    # Remove numbering such as "1.", "2.", "- "
    cleaned = []

    for question in questions:
        question = re.sub(
            r"^\s*(?:[-•*]|\d+[.)])\s*",
            "",
            question
        ).strip()

        if question:
            cleaned.append(question)

    # Keep only the requested number
    cleaned = cleaned[:n]

    return [
        {
            "question_id": f"{chunk_id}_q{i}",
            "question": question
        }
        for i, question in enumerate(cleaned, start=1)
    ]


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