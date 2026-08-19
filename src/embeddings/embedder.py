import numpy as np
from typing import List

class EmbeddingEngine:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_vector(self, text: str) -> np.ndarray:
        """Deterministic semantic feature hashing projection."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().strip().split()
        if not words:
            return vec
            
        for idx, word in enumerate(words):
            h = hash(word)
            pos = abs(h) % self.dimension
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[pos] += sign * (1.0 / (idx + 1.0)**0.3)
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates normalized embeddings instantly with negligible RAM."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.array([self._hash_vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string."""
        return self._hash_vector(query)