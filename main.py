import os
import time
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.chunking.chunker import MultiStrategyChunker
from src.ingestion.loader import load_msmarco_xi
from src.embeddings.embedder import EmbeddingEngine
from src.guardrails.guardrail import SafetyAndRelevanceGuardrail
from src.harness.orchestrator import PipelineHarness, RAGResponse
from src.llm.llm_client import FastLLMClient
from src.retrieval.retriever import DenseRetriever
from src.stt.stt_client import SarvamSTTClient, SarvamSTTError
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

    msmarco_docs = load_msmarco_xi(
        sample_size=100,
    )

    chunker = MultiStrategyChunker()
    chunks = chunker.chunk_all(msmarco_docs, strategy="metadata_aware")
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    vstore.add_documents(vectors, chunks)

    print(
        f"Indexed {len(chunks)} chunks into FAISS from MSMARCO-XI successfully."
    )


def _empty_response(query: str, answer: str, stt_latency_ms: float = 0.0) -> dict:
    """Shared shape for the 'nothing to work with' / 'failed before the
    pipeline ran' cases — always status='error' so the frontend shows it
    as an error rather than a real answer."""
    return {
        "query": query,
        "answer": answer,
        "grounded": False,
        "sources": [],
        "latency": {
            "stt_ms": stt_latency_ms,
            "embedding_ms": 0.0,
            "retrieval_ms": 0.0,
            "guardrail_ms": 0.0,
            "llm_ms": 0.0,
            "total_ms": stt_latency_ms,
        },
        "status": "error",
    }


@app.post("/query")
async def process_query(
    query_text: str = Form(None),
    audio_file: UploadFile = File(None),
):
    stt_latency = 0.0
    final_query = (query_text or "").strip()

    try:
        if audio_file:
            t0 = time.perf_counter()
            audio_bytes = await audio_file.read()
            try:
                final_query = stt_client.transcribe_audio_bytes(
                    audio_bytes, filename=audio_file.filename
                )
            except SarvamSTTError as stt_exc:
                stt_latency = round((time.perf_counter() - t0) * 1000, 2)
                return JSONResponse(
                    status_code=502,
                    content=_empty_response(
                        query="", answer=str(stt_exc), stt_latency_ms=stt_latency
                    ),
                )
            stt_latency = round((time.perf_counter() - t0) * 1000, 2)

        # No fallback question here on purpose — silence, a failed
        # transcription, or an empty text box must surface as an error,
        # never get answered as if the user asked something.
        if not final_query:
            return JSONResponse(
                status_code=400,
                content=_empty_response(
                    query="",
                    answer=(
                        "No speech detected. Please try speaking again, "
                        "or type your question instead."
                    ),
                    stt_latency_ms=stt_latency,
                ),
            )

        response = harness.execute_pipeline(
            final_query, stt_latency_ms=stt_latency
        )
        return response.dict()

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=_empty_response(
                query=final_query,
                answer=f"Handled exception: {str(exc)}",
                stt_latency_ms=stt_latency,
            ),
        )


@app.get("/")
def index_page():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)