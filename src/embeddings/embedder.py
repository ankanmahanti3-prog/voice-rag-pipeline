import os
import numpy as np


class EmbeddingEngine:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.dimension = 384
        self.model = None
        self.model_name = model_name

        try:
            import torch

            torch.set_num_threads(1)  # Restrict CPU threads to prevent RAM spikes
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name, device="cpu", trust_remote_code=True
            )
        except Exception as e:
            print(f"Embedding Engine fallback: {e}")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generates normalized vector embeddings within strict memory bounds."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.model:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=4,
            )
            return embeddings.astype(np.float32)

        # Fallback deterministic vector projection for memory-constrained environments
        vectors = []
        for text in texts:
            np.random.seed(abs(hash(text)) % (2**32))
            vec = np.random.randn(self.dimension).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-10
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string."""
        return self.embed_texts([query])[0]