# Terminal RAG Study Assistant

Terminal only. No server, no frontend, no conversation IDs. Give a PDF path, ask questions.

## Run
```
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
ollama pull embeddinggemma
python main.py                      # start chat
python main.py notes.pdf lectures/  # index PDFs / folders first
```
In chat: `/add <path>`  `/docs`  `/verbose`  `/help`  `/quit`

Delete the old `chroma_db/` folder from the previous version before first run.
First question downloads the reranker model once (needs internet).

## Pipeline
**Index** (`/add`)
1. `document_processor.py`: PyMuPDF text per page -> overlapping chunks, each prefixed `Source: file, Page: n`
2. `llm.py`: Reverse-HyDE, LLM writes 3 questions per chunk
3. `embeddings.py`: embed chunk + questions
4. `vector_db.py`: store in ChromaDB. BM25 index rebuilt from Chroma.

**Query**
1. dense search (ChromaDB, top 20) + BM25 (top 20)
2. Reciprocal Rank Fusion, k=60, keep top 15
3. CrossEncoder rerank, keep top 3
4. Qwen answers from those 3 chunks, streamed, with sources

Use `/verbose` to see each stage's ranking. Tune everything in `config.py`
(`USE_REVERSE_HYDE = False` = much faster indexing, to compare recall).

## Files
| file | job |
|---|---|
| `main.py` | terminal loop, commands, prompt building |
| `document_processor.py` | PDF extraction, chunking |
| `vector_db.py` | ChromaDB, BM25, RRF, rerank |
| `llm.py` | Ollama: answers, HyDE questions, health check |
| `embeddings.py` | Ollama embeddings |
| `config.py` | models, chunk sizes, retrieval params |