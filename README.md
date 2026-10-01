# Terminal RAG Study Assistant

## About the Project

This project was developed to practice the core principles of **Retrieval-Augmented Generation (RAG)** and explore how different retrieval techniques can improve the relevance and accuracy of LLM-generated answers.

The main goal is to understand how combining **dense retrieval, BM25, metadata, Reciprocal Rank Fusion (RRF), CrossEncoder reranking, and Reverse-HyDE** can help an LLM retrieve more relevant context from PDF documents before generating an answer.

The project also provides hands-on experience with document processing, embeddings, vector databases, hybrid search, and local LLM integration.

## Metadata

Metadata is stored alongside the original document chunks and hypothetical questions to preserve document context and maintain traceability.

* **Original chunks:** `source`, `page`, `section`, `title`, `chunk_id`, `path`, `chunk` (chunk index), `question_count`, and `question_ids`.
* **Hypothetical questions:** `path`, `source`, `page`, `section`, `title`, `parent_chunk_id`, `parent_chroma_id`, and `question_id`.

The `parent_chroma_id` field connects each hypothetical question to its original chunk. This allows Reverse-HyDE retrieval results to be mapped back to the actual document content.

## Terminal RAG Study Assistant

Terminal only. No server, no frontend, no conversation IDs. Give a PDF path, ask questions.

## Run

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
ollama pull embeddinggemma
python main.py                      # start chat
python main.py notes.pdf lectures/  # index PDFs / folders first
```

In chat: `/add <path>`  `/docs`  `/verbose`  `/help`  `/quit`

Delete the old `chroma_db/` folder from the previous version before first run.

The first question downloads the reranker model once (requires an internet connection).

## Pipeline

**Index** (`/add`)

1. `document_processor.py`: Extracts text from each PDF page using PyMuPDF, creates overlapping chunks, and prefixes each chunk with `Source: file, Page: n`.
2. `llm.py`: Generates three hypothetical questions per chunk using Reverse-HyDE.
3. `embeddings.py`: Creates embeddings for original chunks and hypothetical questions.
4. `vector_db.py`: Stores original chunks and hypothetical questions in separate ChromaDB collections. Rebuilds the BM25 index from the original chunks.

**Query**

1. **Hybrid retrieval:** Dense search (ChromaDB, top 20) and BM25 (top 20).
2. **Reciprocal Rank Fusion:** Combines dense, BM25, and Reverse-HyDE rankings using `k=60`, keeping the top 15 candidates.
3. **CrossEncoder reranking:** Reorders candidates based on query relevance and keeps the top 3.
4. **Answer generation:** Qwen generates a streamed answer using the three selected original chunks, with sources.

Use `/verbose` to inspect the rankings at each stage. Tune the retrieval parameters in `config.py`.

Set `USE_REVERSE_HYDE = False` to disable Reverse-HyDE and make indexing faster, allowing you to compare retrieval results with and without it.

## Files

| File                    | Job                                                   |
| ----------------------- | ----------------------------------------------------- |
| `main.py`               | Terminal loop, commands, prompt building              |
| `document_processor.py` | PDF extraction, chunking                              |
| `vector_db.py`          | ChromaDB, BM25, RRF, reranking                        |
| `llm.py`                | Ollama: answers, hypothetical questions, health check |
| `embeddings.py`         | Ollama embeddings                                     |
| `config.py`             | Models, chunk sizes, retrieval parameters             |
