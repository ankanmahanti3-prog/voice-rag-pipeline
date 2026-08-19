import re
from typing import List
import numpy as np


class EmbeddingEngine:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocabulary = {}
        self.stopwords = {
            "is",
            "the",
            "a",
            "an",
            "and",
            "or",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "about",
            "tell",
            "me",
            "what",
            "how",
            "who",
            "why",
            "which",
        }

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower())
        return [w for w in words if w not in self.stopwords and len(w) > 1]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Builds indexed semantic vocabulary and encodes document passages."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Index vocabulary dynamically
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                if (
                    token not in self.vocabulary
                    and len(self.vocabulary) < self.dimension
                ):
                    self.vocabulary[token] = len(self.vocabulary)

        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = self._tokenize(text)
            for token in tokens:
                if token in self.vocabulary:
                    vec[self.vocabulary[token]] += 1.0

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            else:
                vec[0] = 1.0
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encodes user queries into the matched document vector space."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(query)

        matched = 0
        for token in tokens:
            if token in self.vocabulary:
                vec[self.vocabulary[token]] += 2.0  # Boost exact keyword matches
                matched += 1

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec