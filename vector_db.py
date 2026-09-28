import chromadb
import config
import os
import pickle
from rank_bm25 import BM25Okapi
from embeddings import get_embedding

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.DB_DIR)
        self.collection = self.client.get_or_create_collection(name="study_materials")
        self.bm25_path = os.path.join(config.DB_DIR, "bm25_index.pkl")
        self.chunks_path = os.path.join(config.DB_DIR, "bm25_chunks.pkl")
        self.bm25 = None
        self.corpus = []
        self._load_bm25()

    def _load_bm25(self):
        if os.path.exists(self.bm25_path) and os.path.exists(self.chunks_path):
            with open(self.bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(self.chunks_path, 'rb') as f:
                self.corpus = pickle.load(f)

    def _save_bm25(self):
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.corpus, f)

    def add_chunks(self, chunks: list[dict], doc_id: str):
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            
            combined_text = f"{chunk['text']}\n\nQuestions:\n{chunk['hypothetical_questions']}"
            documents.append(chunk['text'])
            embeddings.append(get_embedding(combined_text))
            
            meta = chunk['metadata'].copy()
            meta['hypothetical_questions'] = chunk['hypothetical_questions']
            metadatas.append(meta)

            self.corpus.append({
                "id": chunk_id,
                "text": chunk['text'],
                "metadata": meta
            })

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        tokenized_corpus = [doc['text'].split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save_bm25()

    def search(self, query: str, n_results: int = 5):
        if not self.corpus:
            return []

        # Dense search
        query_embedding = get_embedding(query)
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=len(self.corpus)
        )
        dense_ids = dense_results["ids"][0]
        
        # Sparse search (BM25)
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(
            [(self.corpus[i]['id'], score) for i, score in enumerate(bm25_scores)],
            key=lambda x: x[1], reverse=True
        )
        
        # RRF
        k = 60
        rrf_scores = {}
        
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            
        for rank, (doc_id, _) in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            
        # Retrieve top 15 from RRF for reranking
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        
        pre_rerank_chunks = []
        for doc_id, _ in sorted_docs:
            for item in self.corpus:
                if item['id'] == doc_id:
                    pre_rerank_chunks.append(item['text'])
                    break
                    
        if not pre_rerank_chunks:
            return []

        # Rerank
        from sentence_transformers import CrossEncoder
        if not hasattr(self, 'reranker'):
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [[query, chunk] for chunk in pre_rerank_chunks]
        rerank_scores = self.reranker.predict(pairs)
        
        reranked = sorted(zip(pre_rerank_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
        final_chunks = [chunk for chunk, score in reranked[:n_results]]
        
        return final_chunks
