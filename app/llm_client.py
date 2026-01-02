import requests
from app.config import OLLAMA_URL, OLLAMA_MODEL_NAME

def call_llm(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL_NAME) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload)
    return resp.json()["message"]["content"]
