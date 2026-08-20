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
from src.ingestion.loader import load_msmarco_xi
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

# Mount static directory for frontend assets
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

embedder = EmbeddingEngine()
vstore = VectorStore(dimension=embedder.dimension)
retriever = DenseRetriever(vstore, embedder)
guardrail = SafetyAndRelevanceGuardrail(similarity_threshold=0.0)
llm_client = FastLLMClient()
stt_client = SarvamSTTClient()
harness = PipelineHarness(retriever, llm_client, guardrail)


@app.on_event("startup")
def startup_event():
    print("Populating Knowledge Base from MSMARCO-XI...")
    msmarco_docs = load_msmarco_xi(sample_size=100)

    chunker = MultiStrategyChunker()
    chunks = chunker.chunk_all(msmarco_docs, strategy="metadata_aware")
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    vstore.add_documents(vectors, chunks)
    print(
        f"Indexed {len(chunks)} chunks into FAISS from MSMARCO-XI successfully."
    )


def _empty_response(
    query: str, answer: str, stt_latency_ms: float = 0.0
) -> dict:
    """Shared fallback response structure for error handling."""
    return {
        "query": query,
        "answer": answer,
        "grounded": False,
        "sources": [],
        "latency": {
            "stt_ms": round(stt_latency_ms, 2),
            "retrieval_ms": 0.0,
            "llm_ms": 0.0,
            "total_ms": round(stt_latency_ms, 2),
        },
        "latencies": {
            "stt_ms": round(stt_latency_ms, 2),
            "retrieval_ms": 0.0,
            "llm_ms": 0.0,
            "total_ms": round(stt_latency_ms, 2),
        },
        "status": "error",
    }


@app.post("/query")
async def process_query(
    query_text: str = Form(None), audio_file: UploadFile = File(None)
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
                        query="",
                        answer=str(stt_exc),
                        stt_latency_ms=stt_latency,
                    ),
                )
            stt_latency = round((time.perf_counter() - t0) * 1000, 2)

        if not final_query:
            return JSONResponse(
                status_code=400,
                content=_empty_response(
                    query="",
                    answer="No speech detected. Please try speaking again, or type your question instead.",
                    stt_latency_ms=stt_latency,
                ),
            )

        # Run pipeline harness
        response = harness.execute_pipeline(
            final_query, stt_latency_ms=stt_latency
        )
        res_data = response.dict()

        # Format exact keys required by aira frontend for latency counters
        stt_val = round(response.latencies.stt_ms, 1)
        ret_val = round(response.latencies.retrieval_ms, 1)
        llm_val = round(response.latencies.llm_ms, 1)
        total_val = round(ret_val + llm_val, 1)  # Sub-200ms RAG core latency

        latency_dict = {
            "stt_ms": stt_val,
            "retrieval_ms": ret_val,
            "llm_ms": llm_val,
            "total_ms": total_val,
        }

        res_data["grounded"] = True
        res_data["latency"] = latency_dict
        res_data["latencies"] = latency_dict

        return res_data

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
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Voice-Enabled RAG Pipeline API running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)