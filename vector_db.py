import hashlib
import os
import re

import chromadb
from rank_bm25 import BM25Okapi

import config
from embeddings import get_embedding


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25."""
    return re.findall(r"\w+", text.lower())


class VectorDB:
    """Hybrid retrieval: dense (ChromaDB) + sparse (BM25) -> RRF -> cross-encoder rerank.

    ChromaDB is the single source of truth. The BM25 index is rebuilt from it
    on start and after every add/delete, so there are no extra files to keep in sync.
    """

    def __init__(self):
        os.makedirs(config.DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=config.DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
        self.reranker = None
        self.last_trace = {}   # per-stage results of the last search (for /verbose)
        self._rebuild_bm25()

    # ── index management ──────────────────────────────────────────────────────
    def _rebuild_bm25(self):
        data = self.collection.get(include=["documents", "metadatas"])
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.pos = {id_: i for i, id_ in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs]) if self.docs else None

    def count(self) -> int:
        return len(self.ids)

    def list_documents(self) -> list[dict]:
        docs = {}
        for m in self.metas:
            d = docs.setdefault(m["path"], {"path": m["path"], "source": m["source"], "chunks": 0})
            d["chunks"] += 1
        return list(docs.values())

    def has_document(self, path: str) -> bool:
        return any(m["path"] == path for m in self.metas)

    def delete_document(self, path: str):
        self.collection.delete(where={"path": path})
        self._rebuild_bm25()

    def add_chunks(self, chunks: list[dict], path: str, on_progress=None):
        """Embed every chunk (+ its hypothetical questions), then store all at once.
        Embeddings are computed first so a failure leaves the DB untouched."""
        doc_key = hashlib.md5(path.encode()).hexdigest()[:8]  # same filename in two folders = no id clash
        ids, docs, embeddings, metas = [], [], [], []
        for i, chunk in enumerate(chunks):
            questions = chunk.get("questions", "")
            embed_input = chunk["text"] + (f"\n\nQuestions:\n{questions}" if questions else "")
            ids.append(f"{doc_key}_{i}")
            docs.append(chunk["text"])
            embeddings.append(get_embedding(embed_input))
            metas.append({**chunk["metadata"], "path": path, "chunk": i, "questions": questions})
            if on_progress:
                on_progress(i + 1, len(chunks))

        for s in range(0, len(ids), 500):  # Chroma has a max batch size
            self.collection.add(
                ids=ids[s:s + 500], documents=docs[s:s + 500],
                embeddings=embeddings[s:s + 500], metadatas=metas[s:s + 500],
            )
        self._rebuild_bm25()

    # ── retrieval ─────────────────────────────────────────────────────────────
    def _label(self, id_: str) -> str:
        m = self.metas[self.pos[id_]]
        return f"{m['source']} p.{m['page']} #{m['chunk']}"

    def _get_reranker(self):
        if self.reranker is None:
            print("(loading reranker, first run downloads it...)")
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(config.RERANK_MODEL)
        return self.reranker

    def search(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or config.FINAL_TOP_K
        self.last_trace = {}
        if not self.docs:
            return []

        # 1. Dense: embed query, nearest chunks in ChromaDB
        n = min(config.DENSE_TOP_K, len(self.ids))
        res = self.collection.query(query_embeddings=[get_embedding(query)], n_results=n)
        dense_ids = res["ids"][0]

        # 2. Sparse: BM25 keyword scores over all chunks
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:config.BM25_TOP_K]
        bm25_ids = [self.ids[i] for i in ranked if scores[i] > 0]

        # 3. RRF: score = sum over rankings of 1 / (k + rank). Only ranks matter, not raw scores.
        rrf = {}
        for ranking in (dense_ids, bm25_ids):
            for rank, id_ in enumerate(ranking, start=1):
                rrf[id_] = rrf.get(id_, 0.0) + 1.0 / (config.RRF_K + rank)
        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:config.RERANK_TOP_N]

        # 4. Rerank: cross-encoder reads (query, chunk) together = slower but more precise
        pairs = [[query, self.docs[self.pos[id_]]] for id_, _ in fused]
        rerank_scores = self._get_reranker().predict(pairs)
        reranked = sorted(
            zip([id_ for id_, _ in fused], (float(s) for s in rerank_scores)),
            key=lambda x: x[1], reverse=True,
        )[:top_k]

        self.last_trace = {
            "dense":  [(self._label(i), None) for i in dense_ids[:5]],
            "bm25":   [(self._label(i), None) for i in bm25_ids[:5]],
            "rrf":    [(self._label(i), s) for i, s in fused[:5]],
            "rerank": [(self._label(i), s) for i, s in reranked],
        }
        return [
            {"text": self.docs[self.pos[i]], "metadata": self.metas[self.pos[i]], "score": s}
            for i, s in reranked
        ]