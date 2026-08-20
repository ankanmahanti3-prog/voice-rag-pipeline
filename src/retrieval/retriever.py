from typing import Any, Dict, List, Tuple
from src.embeddings.embedder import EmbeddingEngine
from src.vectordb.vector_store import VectorStore


class DenseRetriever:

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingEngine):
        self.vector_store = vector_store
        self.embedder = embedder

        def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
            """Encodes a single user query and fetches top-k closest documents."""
            query_vector = self.embedder.embed_query(query)
            results = self.vector_store.search(
                query_vector.reshape(1, -1),
                top_k=top_k,
            )
            return results

if __name__ == "__main__":
    import numpy as np

    embedder = EmbeddingEngine()
    vstore = VectorStore(dimension=embedder.dimension)

    sample_chunks = [
        {"chunk_id": "c1", "text": "Python is a popular programming language."},
        {
            "chunk_id": "c2",
            "text": "Retrieval Augmented Generation enhances LLM context.",
        },
    ]
    vecs = embedder.embed_texts([c["text"] for c in sample_chunks])
    vstore.add_documents(vecs, sample_chunks)

    retriever = DenseRetriever(vstore, embedder)
    top_matches = retriever.retrieve("What is RAG?", top_k=1)
    print("Retriever Test Result:", top_matches)