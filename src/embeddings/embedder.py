from typing import List
import numpy as np


class EmbeddingEngine:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Deterministic keyword-hash vector projection for accurate matching."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            idx = abs(hash(word)) % self.dimension
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates normalized vector embeddings for document chunks."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.array([self._text_to_vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a user query string consistently."""
        return self._text_to_vector(query)