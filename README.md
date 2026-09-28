# Study Assistant: Advanced RAG Learning Project

## Objective
Project built to learn and implement Advanced Retrieval-Augmented Generation (RAG) techniques. Evaluates local LLMs and embedding models for document Q&A.

## Pipeline Architecture

### 1. Indexing Phase
* **PDF Extraction**: Extracts text, page numbers, and source metadata.
* **Contextual Chunking**: Prefixes metadata (source, page) to chunks to preserve context.
* **Reverse-HyDE**: Uses Qwen2.5 to generate 3 hypothetical questions per chunk during indexing. Shifts computation from query-time to index-time.
* **Embedding**: Embeds combined chunk text and hypothetical questions using Gemma.
* **Storage**: Stores Dense embeddings in ChromaDB and Sparse tokenized text in BM25 index.

### 2. Query & Retrieval Phase
* **Hybrid Search**: Embeds user query. Searches ChromaDB (Dense) and BM25 (Sparse) concurrently.
* **Reciprocal Rank Fusion (RRF)**: Combines Dense and Sparse results mathematically (k=60) to normalize scores.
* **Reranking**: Passes top 15 RRF results through `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) for semantic precision reranking.
* **Generation**: Top 3 reranked chunks fed to Qwen2.5 for final answer synthesis.

## File Breakdown

* `app.py`: FastAPI web server. Manages `/upload` (triggers ingestion) and `/chat` (triggers retrieval/generation) routes.
* `document_processor.py`: Uses `PyMuPDF` (`fitz`). Extracts raw text + metadata. Splits text into contextualized chunks.
* `vector_db.py`: Core retrieval engine. Manages ChromaDB client, BM25 pickling, Hybrid Search logic, RRF computation, and CrossEncoder reranking.
* `llm.py`: Interfaces with Ollama API for Qwen2.5. Handles answer streaming and Reverse-HyDE question generation.
* `embeddings.py`: Interfaces with Ollama API for Gemma embeddings.
* `ingest.py`: Standalone CLI script for PDF ingestion pipeline.
* `chat.py`: Standalone CLI script for interactive terminal chat.
* `config.py`: Centralized configuration (model names, chunk sizes, DB paths).
* `requirements.txt`: Python dependencies (`fastapi`, `chromadb`, `rank_bm25`, `sentence-transformers`, `pymupdf`).
