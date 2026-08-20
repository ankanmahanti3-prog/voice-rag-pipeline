from typing import List
import numpy as np


class EmbeddingEngine:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = [w for w in text.lower().replace("-", " ").split() if len(w) > 1]
        if not tokens:
            vec[0] = 1.0
            return vec

        for pos, token in enumerate(tokens):
            h = 0
            for ch in token:
                h = (h * 33 + ord(ch)) & 0xFFFFFFFF
            idx = h % self.dimension
            vec[idx] += 1.0 + (1.0 / (pos + 1))

        norm = np.linalg.norm(vec)
        return (
            (vec / norm)
            if norm > 0
            else np.ones(self.dimension, dtype=np.float32)
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.array([self._hash_vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._hash_vector(query)