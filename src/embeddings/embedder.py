from typing import List
import numpy as np


class EmbeddingEngine:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocab = {}

    def _tokenize(self, text: str) -> List[str]:
        cleaned = "".join(
            [c.lower() if c.isalnum() else " " for c in text]
        ).strip()
        return [w for w in cleaned.split() if len(w) > 1]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates normalized TF-IDF semantic embeddings without heavy PyTorch dependencies."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Build / expand dynamic vocabulary
        for text in texts:
            for word in self._tokenize(text):
                if word not in self.vocab and len(self.vocab) < self.dimension:
                    self.vocab[word] = len(self.vocab)

        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = self._tokenize(text)
            for word in tokens:
                if word in self.vocab:
                    vec[self.vocab[word]] += 1.0

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            else:
                vec[0] = 1.0  # Safe unit vector
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string against the existing indexed vocabulary."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(query)
        for word in tokens:
            if word in self.vocab:
                vec[self.vocab[word]] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec