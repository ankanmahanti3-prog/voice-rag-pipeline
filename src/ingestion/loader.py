import logging
from typing import Any, Dict, List
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_msmarco_xi(
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "train",
    sample_size: int = 1000,
) -> List[Dict[str, Any]]:
    """Loads a partition of the MSMARCO-XI dataset for the RAG pipeline."""
    logger.info(
        f"Downloading/loading dataset '{dataset_name}' (split: {split}, size: {sample_size})..."
    )
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    documents = []
    for idx, item in enumerate(dataset):
        if idx >= sample_size:
            break

        doc = {
            "id": item.get("passage_id", str(idx)),
            "text": item.get("passage_text", "") or item.get("passage", ""),
            "metadata": {
                "query": item.get("query", ""),
                "lang": item.get("language", "en"),
                "source_id": item.get("passage_id", str(idx)),
            },
        }
        if doc["text"].strip():
            documents.append(doc)

    logger.info(f"Successfully loaded {len(documents)} document passages.")
    return documents


if __name__ == "__main__":
    docs = load_msmarco_xi(sample_size=10)
    print(f"Sample loaded document:\n{docs[0] if docs else 'No documents'}")