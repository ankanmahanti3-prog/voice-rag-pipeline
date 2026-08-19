from typing import Any, Dict, List
import re


class MultiStrategyChunker:

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def fixed_size_chunking(
        self, doc: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Splits document text using sliding window token/word overlap."""
        words = doc["text"].split()
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_fixed_{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "strategy": "fixed_size_overlap",
                        "start_idx": start,
                        "end_idx": min(end, len(words)),
                    },
                }
            )
            chunk_idx += 1
            start += self.chunk_size - self.chunk_overlap
            if end >= len(words):
                break

        return chunks

    def semantic_sentence_chunking(
        self, doc: Dict[str, Any], max_chunk_chars: int = 600
    ) -> List[Dict[str, Any]]:
        """Splits text along sentence boundaries to preserve complete semantic units."""
        sentences = re.split(r"(?<=[.!?|।])\s+", doc["text"])
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if (
                current_length + len(sentence) > max_chunk_chars
                and current_chunk
            ):
                chunks.append(
                    {
                        "chunk_id": f"{doc['id']}_semantic_{chunk_idx}",
                        "text": " ".join(current_chunk),
                        "metadata": {
                            **doc.get("metadata", {}),
                            "strategy": "semantic_sentence",
                        },
                    }
                )
                chunk_idx += 1
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_semantic_{chunk_idx}",
                    "text": " ".join(current_chunk),
                    "metadata": {
                        **doc.get("metadata", {}),
                        "strategy": "semantic_sentence",
                    },
                }
            )

        return chunks

    def metadata_aware_chunking(
        self, doc: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Embeds query metadata directly into chunk headers for higher retrieval accuracy."""
        base_chunks = self.semantic_sentence_chunking(doc)
        enriched_chunks = []

        query = doc.get("metadata", {}).get("query", "")
        lang = doc.get("metadata", {}).get("lang", "en")

        for chunk in base_chunks:
            header = f"[Context: Query='{query}' | Lang='{lang}']\n"
            enriched_chunks.append(
                {
                    "chunk_id": f"{chunk['chunk_id']}_meta",
                    "text": header + chunk["text"],
                    "metadata": {
                        **chunk["metadata"],
                        "strategy": "metadata_aware",
                    },
                }
            )

        return enriched_chunks

    def chunk_all(
        self, docs: List[Dict[str, Any]], strategy: str = "metadata_aware"
    ) -> List[Dict[str, Any]]:
        """Processes a list of documents using the chosen chunking strategy."""
        all_chunks = []
        for doc in docs:
            if strategy == "fixed":
                all_chunks.extend(self.fixed_size_chunking(doc))
            elif strategy == "semantic":
                all_chunks.extend(self.semantic_sentence_chunking(doc))
            else:
                all_chunks.extend(self.metadata_aware_chunking(doc))
        return all_chunks


if __name__ == "__main__":
    sample_doc = {
        "id": "sample_1",
        "text": "Retrieval Augmented Generation reduces hallucination. It fetches passages from a vector database. Then an LLM creates an accurate answer.",
        "metadata": {"query": "How does RAG work?", "lang": "en"},
    }
    chunker = MultiStrategyChunker(chunk_size=10, chunk_overlap=3)
    chunks = chunker.metadata_aware_chunking(sample_doc)
    print(f"Generated {len(chunks)} chunks with metadata:")
    for c in chunks:
        print(f"\n- ID: {c['chunk_id']}\n  Text: {c['text']}")