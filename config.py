import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── Models ────────────────────────────────────────────────────────────────────
OLLAMA_API_URL = "http://localhost:11434/api"
LLM_MODEL = "qwen2.5:7b-instruct"          # answers + hypothetical questions
EMBEDDING_MODEL = "embeddinggemma:latest"  # dense embeddings
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # downloaded once from HuggingFace

# ── Indexing ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # characters shared between neighbouring chunks
USE_REVERSE_HYDE = True  # LLM writes questions per chunk at index time (slow, better recall)
HYDE_NUM_QUESTIONS = 3

# ── Retrieval ─────────────────────────────────────────────────────────────────
DENSE_TOP_K = 20    # candidates from vector search
BM25_TOP_K = 20     # candidates from keyword search
RRF_K = 60          # Reciprocal Rank Fusion constant
RERANK_TOP_N = 15   # fused candidates sent to the cross-encoder
FINAL_TOP_K = 3     # chunks given to the LLM