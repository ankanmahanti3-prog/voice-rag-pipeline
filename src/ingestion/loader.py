import logging
from typing import Any, Dict, List
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_msmarco_xi(
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    config: str = "hi",
    split: str = "train",
    sample_size: int = 100,
) -> List[Dict[str, Any]]:
    """Loads a small streaming sample from MSMARCO-XI."""

    logger.info(
        f"Streaming dataset '{dataset_name}' "
        f"(config: {config}, split: {split}, size: {sample_size})..."
    )

    dataset = load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=True,
    )

    documents = []

    for idx, item in enumerate(dataset):
        if idx >= sample_size:
            break

        passages = item.get("passages", {})

        translated_passages = passages.get(
            "Translated_passages", []
        )

        english_passages = passages.get(
            "English_passages", []
        )

        # Prefer translated passages.
        passage_list = translated_passages or english_passages

        for passage_idx, text in enumerate(passage_list):
            if not text or not text.strip():
                continue

            documents.append(
                {
                    "id": f"{item.get('query_id', idx)}-{passage_idx}",
                    "text": text.strip(),
                    "metadata": {
                        "query": item.get("query", ""),
                        "lang": config,
                        "source_id": item.get(
                            "query_id",
                            str(idx),
                        ),
                    },
                }
            )

            if len(documents) >= sample_size:
                break

        if len(documents) >= sample_size:
            break

    logger.info(
        f"Successfully loaded {len(documents)} document passages."
    )

    return documents

if __name__ == "__main__":
    docs = load_msmarco_xi(
        config="hi",
        sample_size=10,
    )

    print(
        f"Sample loaded document:\n"
        f"{docs[0] if docs else 'No documents'}"
    )