from typing import Any, Dict, List
import numpy as np


class DenseRetriever:

    def __init__(self, vector_store, embedding_engine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Standard retrieve method expected by orchestrator."""
        return self.retrieve_relevant_chunks(query, top_k=top_k)

    def retrieve_relevant_chunks(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Embeds query, searches vector store, and standardizes output to dictionaries."""
        if not query or not str(query).strip():
            return []

        query_vector = self.embedding_engine.embed_query(query)
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        raw_results = self.vector_store.search(query_vector, top_k=top_k)

        # Standardize results into [{"chunk": chunk_data, "score": score}]
        formatted_results = []
        for item in raw_results:
            if isinstance(item, tuple) and len(item) >= 2:
                chunk, score = item[0], item[1]
                formatted_results.append(
                    {"chunk": chunk, "score": float(score)}
                )
            elif isinstance(item, dict):
                formatted_results.append(item)

        return formatted_results