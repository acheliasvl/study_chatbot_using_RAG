import requests
import json
import config

def generate_answer_stream(prompt: str):
    url = f"{config.OLLAMA_API_URL}/generate"
    payload = {
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            word = chunk.get("response", "")
            yield word

def generate_answer(prompt: str) -> str:
    answer = ""
    for word in generate_answer_stream(prompt):
        print(word, end="", flush=True)
        answer += word
    print()
    return answer
