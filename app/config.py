from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

OLLAMA_MODEL_NAME = "llama3:8b"
OLLAMA_URL = "http://localhost:11434"
