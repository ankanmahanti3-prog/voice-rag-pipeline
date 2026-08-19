import numpy as np
from typing import List

class EmbeddingEngine:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Normalized character-ngram hashing projection for robust semantic matching."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        cleaned = "".join([c.lower() if c.isalnum() else " " for c in text]).strip()
        words = cleaned.split()
        if not words:
            vec[0] = 1.0
            return vec

        # Extract word tokens and character n-grams
        tokens = list(words)
        for w in words:
            if len(w) >= 3:
                tokens.extend([w[i:i+3] for i in range(len(w) - 2)])

        for token in tokens:
            # Deterministic hash independent of python seed
            h = 0
            for char in token:
                h = (h * 31 + ord(char)) & 0xFFFFFFFF
            idx = h % self.dimension
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates normalized embeddings for chunks."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.array([self._text_to_vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds query into the same normalized vector space."""
        return self._text_to_vector(query)