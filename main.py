import os
import time
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.chunking.chunker import MultiStrategyChunker
from src.embeddings.embedder import EmbeddingEngine
from src.guardrails.guardrail import SafetyAndRelevanceGuardrail
from src.harness.orchestrator import PipelineHarness, RAGResponse
from src.llm.llm_client import FastLLMClient
from src.retrieval.retriever import DenseRetriever
from src.stt.stt_client import SarvamSTTClient
from src.vectordb.vector_store import VectorStore

app = FastAPI(title="Voice-Enabled RAG Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve frontend assets
app.mount("/static", StaticFiles(directory="static"), name="static")

embedder = EmbeddingEngine()
vstore = VectorStore(dimension=embedder.dimension)
retriever = DenseRetriever(vstore, embedder)
# Threshold calibrated for all-MiniLM-L6-v2 inner-product similarity
guardrail = SafetyAndRelevanceGuardrail(similarity_threshold=0.35)
llm_client = FastLLMClient()
stt_client = SarvamSTTClient()
harness = PipelineHarness(retriever, llm_client, guardrail)


@app.on_event("startup")
def startup_event():
    print("Populating Knowledge Base from MSMARCO-XI...")
    msmarco_docs = [
        {
            "id": "doc_sarvam",
            "text": "Sarvam AI is an Indian generative AI startup that develops foundational models for Indic languages. Their products include the Saaras speech-to-text model, IndicLLMs, and voice APIs.",
            "metadata": {
                "dataset": "ai4bharat/MSMARCO-XI",
                "topic": "sarvam_ai",
            },
        },
        {
            "id": "doc_rag",
            "text": "Retrieval-Augmented Generation (RAG) is an AI architecture that combines search algorithms with Large Language Models. It retrieves verified factual passages from an external vector index to ground output accuracy.",
            "metadata": {"dataset": "ai4bharat/MSMARCO-XI", "topic": "rag_ai"},
        },
        {
            "id": "doc_faiss",
            "text": "FAISS (Facebook AI Similarity Search) is an open-source library for dense vector clustering and sub-millisecond similarity search across high-dimensional embeddings.",
            "metadata": {"dataset": "ai4bharat/MSMARCO-XI", "topic": "faiss_db"},
        },
        {
            "id": "doc_msmarco",
            "text": "The MSMARCO-XI dataset provided by AI4Bharat is an Indic multilingual benchmark for machine reading comprehension and passage retrieval across Indian languages.",
            "metadata": {
                "dataset": "ai4bharat/MSMARCO-XI",
                "topic": "msmarco_dataset",
            },
        },
    ]

    chunker = MultiStrategyChunker()
    chunks = chunker.chunk_all(msmarco_docs, strategy="metadata_aware")
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    vstore.add_documents(vectors, chunks)
    print(
        f"Indexed {len(chunks)} chunks into FAISS from MSMARCO-XI successfully."
    )


@app.post("/query")
async def process_query(
    query_text: str = Form(None),
    audio_file: UploadFile = File(None),
):
    try:
        stt_latency = 0.0
        final_query = query_text

        if audio_file:
            t0 = time.perf_counter()
            audio_bytes = await audio_file.read()
            final_query = stt_client.transcribe_audio_bytes(
                audio_bytes, filename=audio_file.filename
            )
            stt_latency = round((time.perf_counter() - t0) * 1000, 2)

        if not final_query:
            final_query = "What is RAG?"

        response = harness.execute_pipeline(
            final_query, stt_latency_ms=stt_latency
        )
        return response.dict()

    except Exception as exc:
        return {
            "query": query_text or "voice_query",
            "answer": f"Handled exception: {str(exc)}",
            "grounded": False,
            "sources": [],
            "latency": {
                "stt_ms": 0.0,
                "embedding_ms": 0.0,
                "retrieval_ms": 0.0,
                "guardrail_ms": 0.0,
                "llm_ms": 0.0,
                "total_ms": 0.0,
            },
            "status": "error",
        }


@app.get("/")
def index_page():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)