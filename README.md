# 🎙️ Voice-Enabled Indic RAG Pipeline (<200ms Target)

An ultra-low latency Voice RAG system integrating Sarvam AI (Saaras STT), FAISS Vector Search, MiniLM Embeddings, and Groq (Llama 3 inference).

---

## 🏗️ Architecture Components
- **STT Engine:** Sarvam AI (`saaras:v3`) for low-latency Indic speech-to-text.
- **Embeddings & Vector Store:** SentenceTransformers (`all-MiniLM-L6-v2`) with FAISS IndexFlatIP.
- **LLM Inference:** Groq LPU (`llama-3.1-8b-instant`) for fast response generation.
- **Chunking Strategies:** Sliding-window overlap, semantic delimiter, and metadata-aware prefixing.
- **Guardrails:** Cosine similarity thresholding to block hallucinations and sanitization to catch prompt injections.

---

## 🚀 How to Run
1. Start the web server:
   ```bash
   python main.py