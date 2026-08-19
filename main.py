import os
import time
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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

embedder = EmbeddingEngine()
vstore = VectorStore(dimension=embedder.dimension)
retriever = DenseRetriever(vstore, embedder)
# Threshold calibrated for all-MiniLM-L6-v2 inner-product similarity
guardrail = SafetyAndRelevanceGuardrail(similarity_threshold=0.30)
llm_client = FastLLMClient()
stt_client = SarvamSTTClient()
harness = PipelineHarness(retriever, llm_client, guardrail)


@app.on_event("startup")
def startup_event():
    print("Populating Knowledge Base...")
    sample_docs = [
        {
            "id": "doc_rag",
            "text": "Retrieval-Augmented Generation (RAG) is an AI architecture that combines search algorithms with Large Language Models. It retrieves relevant factual passages from an external vector index and passes them as context to prevent hallucination.",
            "metadata": {
                "query": "What is RAG and how does it work?",
                "lang": "en",
            },
        },
        {
            "id": "doc_sarvam",
            "text": "Sarvam AI is an Indian generative AI startup that develops foundational models for Indic languages. Their products include Saaras speech-to-text models and text-to-speech engines optimized for low latency.",
            "metadata": {"query": "Tell me about Sarvam AI", "lang": "en"},
        },
        {
            "id": "doc_faiss",
            "text": "FAISS (Facebook AI Similarity Search) is an open-source library for dense vector clustering and similarity search. It allows rapid search in high-dimensional vector spaces.",
            "metadata": {"query": "What is FAISS?", "lang": "en"},
        },
        {
            "id": "doc_latency",
            "text": "Achieving sub-200ms latency in voice RAG pipelines requires parallelized audio streaming, in-memory vector stores, and ultra-fast inference APIs like Groq.",
            "metadata": {
                "query": "How to achieve low latency in RAG?",
                "lang": "en",
            },
        },
        {
            "id": "doc_sports",
            "text": "In the Tokyo 2020 Olympics (held in 2021), Marcell Jacobs won the Men's 100m gold medal, and Neeraj Chopra won the Men's Javelin gold medal.",
            "metadata": {
                "query": "Who won Olympic gold medals in running and athletics?",
                "lang": "en",
            },
        },
    ]

    chunker = MultiStrategyChunker()
    chunks = chunker.chunk_all(sample_docs, strategy="metadata_aware")
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    vstore.add_documents(vectors, chunks)
    print(f"Indexed {len(chunks)} chunks into FAISS successfully.")


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


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Voice-Enabled RAG System</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; padding: 40px 20px; }
            .container { max-width: 700px; width: 100%; background: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
            h1 { font-size: 1.4rem; color: #38bdf8; margin-bottom: 8px; }
            p { color: #94a3b8; font-size: 0.9rem; margin-bottom: 20px; }
            .record-box { display: flex; gap: 10px; margin-bottom: 20px; }
            button { background: #0284c7; color: #fff; border: none; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-weight: bold; }
            button:hover { background: #0369a1; }
            button.recording { background: #ef4444; animation: pulse 1s infinite; }
            input[type="text"] { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #fff; }
            .output { background: #0f172a; border-radius: 8px; padding: 16px; border: 1px solid #334155; min-height: 120px; white-space: pre-wrap; font-size: 0.95rem; line-height: 1.5; }
            .metrics { display: flex; gap: 15px; margin-top: 15px; font-size: 0.8rem; color: #38bdf8; font-family: monospace; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Voice-Enabled RAG Pipeline</h1>
            <p>Target Latency: &lt;200ms | Multi-Strategy Chunking & Indic Guardrails</p>
            <div class="record-box">
                <input type="text" id="queryInput" placeholder="Speak or type your question...">
                <button id="recBtn">🎤 Speak</button>
                <button id="sendBtn">Send</button>
            </div>
            <div class="output" id="resultBox">System ready. Click 'Speak' or type to test.</div>
            <div class="metrics" id="latencyBox">Latency: -- ms | STT: -- ms | Retrieval: -- ms | LLM: -- ms</div>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            const recBtn = document.getElementById('recBtn');
            const sendBtn = document.getElementById('sendBtn');
            const queryInput = document.getElementById('queryInput');
            const resultBox = document.getElementById('resultBox');
            const latencyBox = document.getElementById('latencyBox');

            recBtn.onclick = async () => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    recBtn.classList.remove('recording');
                    recBtn.innerText = '🎤 Speak';
                } else {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const formData = new FormData();
                        formData.append('audio_file', audioBlob, 'mic.wav');
                        sendQuery(formData);
                    };
                    mediaRecorder.start();
                    recBtn.classList.add('recording');
                    recBtn.innerText = '⏹ Stop';
                }
            };

            sendBtn.onclick = () => {
                const text = queryInput.value.trim();
                if (!text) return;
                const formData = new FormData();
                formData.append('query_text', text);
                sendQuery(formData);
            };

            async function sendQuery(formData) {
                resultBox.innerText = "Processing pipeline...";
                try {
                    const res = await fetch('/query', { method: 'POST', body: formData });
                    const data = await res.json();
                    resultBox.innerText = `Query: ${data.query}\n\nAnswer: ${data.answer}\n\nStatus: ${data.status}`;
                    latencyBox.innerText = `Total: ${data.latency.total_ms}ms | STT: ${data.latency.stt_ms}ms | Retr: ${data.latency.retrieval_ms}ms | LLM: ${data.latency.llm_ms}ms`;
                } catch(err) {
                    resultBox.innerText = "Error executing pipeline: " + err;
                }
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)