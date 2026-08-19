from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initializes a local embedding model optimized for CPU latency."""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(
        self, texts: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """Converts strings into normalized floating-point vector arrays."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")


if __name__ == "__main__":
    engine = EmbeddingEngine()
    test_vec = engine.embed_texts("Voice-to-text RAG test query")
    print(f"Embedding shape: {test_vec.shape}, Dim: {engine.dimension}")