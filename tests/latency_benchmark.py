import time
import numpy as np
from src.embeddings.embedder import EmbeddingEngine
from src.guardrails.guardrail import SafetyAndRelevanceGuardrail
from src.harness.orchestrator import PipelineHarness
from src.llm.llm_client import FastLLMClient
from src.vectordb.vector_store import VectorStore


def run_latency_benchmark(num_queries: int = 50):
    print("Setting up benchmark test harness...")

    embedder = EmbeddingEngine()
    vstore = VectorStore(dimension=embedder.dimension)
    guardrail = SafetyAndRelevanceGuardrail()
    llm = FastLLMClient()

    sample_docs = [
        {
            "chunk_id": "doc_1",
            "text": "Retrieval Augmented Generation integrates external knowledge sources.",
        },
        {
            "chunk_id": "doc_2",
            "text": "Sarvam AI builds speech and language models tailored for Indic languages.",
        },
        {
            "chunk_id": "doc_3",
            "text": "FAISS provides fast similarity search on dense vector embeddings.",
        },
    ]
    doc_vectors = embedder.embed_texts([d["text"] for d in sample_docs])
    vstore.add_documents(doc_vectors, sample_docs)

    from src.retrieval.retriever import DenseRetriever

    retriever = DenseRetriever(vstore, embedder)
    harness = PipelineHarness(retriever, llm, guardrail)

    benchmark_queries = [
        "What is Retrieval Augmented Generation?",
        "How does FAISS help in search?",
        "Tell me about Sarvam AI models.",
        "How fast is vector search?",
    ] * (num_queries // 4 + 1)

    benchmark_queries = benchmark_queries[:num_queries]

    latencies = []
    print(f"\nRunning {num_queries} pipeline evaluation queries...")

    for i, q in enumerate(benchmark_queries):
        resp = harness.execute_pipeline(
            q, stt_latency_ms=35.0
        )  # 35ms avg streaming STT overhead
        latencies.append(resp.latency.total_ms)

    latencies = np.array(latencies)

    p50 = np.percentile(latencies, 50)
    p70 = np.percentile(latencies, 70)
    p100 = np.percentile(latencies, 100)
    mean_lat = np.mean(latencies)

    print("\n" + "=" * 45)
    print("      PIPELINE LATENCY ANALYTICS REPORT      ")
    print("=" * 45)
    print(f"Total Test Runs : {num_queries}")
    print(f"Mean Latency   : {mean_lat:.2f} ms")
    print(f"P50 (Median)   : {p50:.2f} ms")
    print(f"P70 Latency    : {p70:.2f} ms")
    print(f"P100 (Max)     : {p100:.2f} ms")
    print(
        f"Target (<200ms): {'PASSED' if p70 < 200 else 'NEEDS OPTIMIZATION'}"
    )
    print("=" * 45)


if __name__ == "__main__":
    run_latency_benchmark(num_queries=20)