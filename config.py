import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Models (Ollama)
LLM_MODEL = "qwen2.5:7b-instruct"
EMBEDDING_MODEL = "embeddinggemma:latest"
OLLAMA_API_URL = "http://localhost:11434/api"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure directories exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
