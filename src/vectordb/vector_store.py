import json
import os
from typing import Any, Dict, List, Tuple
import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Use IndexFlatIP (Inner Product) for normalized cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_documents(
        self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]
    ) -> None:
        """Adds dense vectors and their corresponding chunk metadata to the store."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Mismatch between number of embeddings and chunks metadata."
            )
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Finds the top-K nearest neighbors and returns them with similarity scores."""
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        return results

    def save(self, index_file: str, metadata_file: str) -> None:
        """Persists the FAISS index and metadata dictionary to disk."""
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        faiss.write_index(self.index, index_file)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_file: str, metadata_file: str) -> None:
        """Loads a pre-built index and metadata from disk."""
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Missing {index_file} or {metadata_file}")
        self.index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)


if __name__ == "__main__":
    vs = VectorStore(dimension=4)
    dummy_vecs = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype="float32"
    )
    norm_vecs = dummy_vecs / np.linalg.norm(dummy_vecs, axis=1, keepdims=True)
    dummy_chunks = [
        {"chunk_id": "c1", "text": "First passage"},
        {"chunk_id": "c2", "text": "Second passage"},
    ]
    vs.add_documents(norm_vecs, dummy_chunks)
    query_vec = np.array([[0.1, 0.2, 0.3, 0.4]], dtype="float32")
    query_vec = query_vec / np.linalg.norm(query_vec)
    res = vs.search(query_vec, top_k=2)
    print(f"Sample search results: {res}")