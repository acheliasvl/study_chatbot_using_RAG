import chromadb
import config
from embeddings import get_embedding

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.DB_DIR)
        self.collection = self.client.get_or_create_collection(name="study_materials")

    def add_chunks(self, chunks: list[str], doc_id: str):
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = [get_embedding(chunk) for chunk in chunks]
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

    def search(self, query: str, n_results: int = 5):
        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results["documents"][0]
