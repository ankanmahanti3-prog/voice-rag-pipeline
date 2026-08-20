from typing import Any, Dict, List
import numpy as np


class DenseRetriever:

    def __init__(self, vector_store, embedding_engine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Standard retrieve method expected by the harness."""
        return self.retrieve_relevant_chunks(query, top_k=top_k)

    def retrieve_relevant_chunks(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Embeds query and searches FAISS vector store with fallback handling."""
        if not query or not query.strip():
            return []

        query_vector = self.embedding_engine.embed_query(query)
        # Ensure correct 2D shape for FAISS inner-product
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        results = self.vector_store.search(query_vector, top_k=top_k)
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