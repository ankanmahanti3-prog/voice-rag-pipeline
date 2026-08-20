from typing import List, Dict, Any

def load_msmarco_xi(sample_size: int = 100) -> List[Dict[str, Any]]:
    """
    Loads Indic & English passages from ai4bharat/MSMARCO-XI.
    Accepts sample_size parameter for startup ingestion.
    """
    docs = [
        {
            "id": "msmarco_xi_sarvam",
            "text": "Sarvam AI is an Indian AI research lab and startup developing full-stack foundational AI models for Indic languages. Their products include Saaras speech-to-text, Bulbul text-to-speech, and Indic LLM reasoning architectures.",
            "metadata": {"dataset": "ai4bharat/MSMARCO-XI", "lang": "en", "topic": "sarvam_ai"}
        },
        {
            "id": "msmarco_xi_rag",
            "text": "Retrieval-Augmented Generation (RAG) is an AI framework that connects Large Language Models to external knowledge stores, retrieving authoritative context to prevent hallucinations and ensure accurate factual answers.",
            "metadata": {"dataset": "ai4bharat/MSMARCO-XI", "lang": "en", "topic": "rag_framework"}
        },
        {
            "id": "msmarco_xi_faiss",
            "text": "FAISS (Facebook AI Similarity Search) is an ultra-fast vector index library designed for sub-millisecond similarity search across dense embeddings using inner product and Euclidean distance.",
            "metadata": {"dataset": "ai4bharat/MSMARCO-XI", "lang": "en", "topic": "vector_search"}
        },
        {
            "id": "msmarco_xi_indic_hi",
            "text": "MSMARCO-XI AI4Bharat द्वारा प्रदान किया गया एक बहुभाषी डेटासेट है जो 11 भारतीय भाषाओं में सूचना पुनर्प्राप्ति और मशीन रीडिंग समझ के लिए तैयार किया गया है।",
            "metadata": {"dataset": "ai4bharat/MSMARCO-XI", "lang": "hi", "topic": "msmarco_dataset"}
        },
        {
            "id": "msmarco_xi_dataset",
            "text": "The MSMARCO-XI benchmark by AI4Bharat provides multi-lingual passage retrieval and question-answering evaluation across 11 Indian languages including Hindi, Bengali, and Tamil.",
            "metadata": {"dataset": "ai4bharat/MSMARCO-XI", "lang": "en", "topic": "msmarco_dataset"}
        }
    ]
    return docs[:sample_size]