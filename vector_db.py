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
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        self.question_collection = self.client.get_or_create_collection(
                name="hypothetical_questions",
                metadata={"hnsw:space": "cosine"}
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
        self.question_collection.delete(where={"path": path})
        self._rebuild_bm25()

    def add_chunks(self, chunks: list[dict], path: str, on_progress=None):
        """
        Store original chunks and create separate Reverse-HyDE vectors
        for each hypothetical question.

        Each hypothetical question keeps an explicit mapping to its
        parent chunk through parent_chunk_id.
        """

        doc_key = hashlib.md5(path.encode()).hexdigest()[:8]

        # -------------------------
        # Original chunk records
        # -------------------------

        ids = []
        docs = []
        embeddings = []
        metas = []

        # -------------------------
        # Reverse-HyDE records
        # -------------------------

        question_ids = []
        question_texts = []
        question_embeddings = []
        question_metas = []

        for i, chunk in enumerate(chunks):

            chunk_id = chunk["metadata"].get("chunk_id", i)

            chunk_chroma_id = f"{doc_key}_{chunk_id}"

            # Original chunk vector
            ids.append(chunk_chroma_id)
            docs.append(chunk["text"])
            embeddings.append(
                get_embedding(chunk["text"])
            )

            questions = chunk.get("questions", [])

            metas.append({
                **chunk["metadata"],
                "path": path,
                "chunk": i,
                "question_count": len(questions),
                "question_ids": ",".join(
                    q["question_id"] for q in questions
                )
            })

            # -------------------------
            # Individual question vectors
            # -------------------------

            for q in questions:

                question_id = q["question_id"]
                question = q["question"]

                question_ids.append(
                    f"{doc_key}_{question_id}"
                )

                question_texts.append(question)

                question_embeddings.append(
                    get_embedding(question)
                )

                question_metas.append({
                    "path": path,
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "section": chunk["metadata"]["section"],
                    "title": chunk["metadata"]["title"],
                    "parent_chunk_id": str(chunk_id),
                    "parent_chroma_id": chunk_chroma_id,
                    "question_id": question_id,
                })

            if on_progress:
                on_progress(
                    i + 1,
                    len(chunks)
                )

        # -------------------------
        # Store original chunks
        # -------------------------

        for s in range(0, len(ids), 500):

            self.collection.add(
                ids=ids[s:s + 500],
                documents=docs[s:s + 500],
                embeddings=embeddings[s:s + 500],
                metadatas=metas[s:s + 500],
            )

        # -------------------------
        # Store hypothetical questions
        # -------------------------

        for s in range(0, len(question_ids), 500):

            self.question_collection.add(
                ids=question_ids[s:s + 500],
                documents=question_texts[s:s + 500],
                embeddings=question_embeddings[s:s + 500],
                metadatas=question_metas[s:s + 500],
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

        # 1. Embed the query once
        query_embedding = get_embedding(query)

        # 2. Dense retrieval
        n = min(config.DENSE_TOP_K, len(self.ids))

        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n
        )
        dense_ids = res["ids"][0]

        # 3. Reverse-HyDE question retrieval
        question_n = min(
            config.DENSE_TOP_K,
            self.question_collection.count()
        )

        if question_n > 0:
            question_res = self.question_collection.query(
                query_embeddings=[query_embedding],
                n_results=question_n
            )
            question_results = question_res["metadatas"][0]
        else:
            question_results = []

        # Map questions to unique original chunks
        hyde_ids = []
        seen = set()

        for meta in question_results:
            parent_id = meta["parent_chroma_id"]

            if parent_id in self.pos and parent_id not in seen:
                hyde_ids.append(parent_id)
                seen.add(parent_id)

        # 4. Sparse retrieval: BM25
        scores = self.bm25.get_scores(tokenize(query))

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:config.BM25_TOP_K]

        bm25_ids = [
            self.ids[i]
            for i in ranked
            if scores[i] > 0
        ]

        # 5. RRF fusion
        rrf = {}

        for ranking in (dense_ids, bm25_ids, hyde_ids):
            for rank, id_ in enumerate(ranking, start=1):
                rrf[id_] = rrf.get(id_, 0.0) + (
                    1.0 / (config.RRF_K + rank)
                )

        fused = sorted(
            rrf.items(),
            key=lambda x: x[1],
            reverse=True
        )[:config.RERANK_TOP_N]

        # 6. Retrieve original chunk texts for reranking
        candidates = [
            (id_, self.docs[self.pos[id_]])
            for id_, _ in fused
        ]

        # 7. CrossEncoder reranking
        pairs = [
            [query, text]
            for _, text in candidates
        ]

        if pairs:
            rerank_scores = self._get_reranker().predict(pairs)

            reranked = sorted(
                zip(
                    [id_ for id_, _ in candidates],
                    (float(s) for s in rerank_scores)
                ),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
        else:
            reranked = []

        # 8. Save retrieval trace for /verbose
        self.last_trace = {
            "dense": [
                (self._label(i), None)
                for i in dense_ids[:5]
            ],
            "bm25": [
                (self._label(i), None)
                for i in bm25_ids[:5]
            ],
            "hyde": [
                (self._label(i), None)
                for i in hyde_ids[:5]
            ],
            "rrf": [
                (self._label(i), s)
                for i, s in fused[:5]
            ],
            "rerank": [
                (self._label(i), s)
                for i, s in reranked
            ],
        }

        # 9. Return final chunks
        return [
            {
                "text": self.docs[self.pos[i]],
                "metadata": self.metas[self.pos[i]],
                "score": s
            }
            for i, s in reranked
        ]