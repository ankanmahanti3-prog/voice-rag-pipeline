import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.guardrails.guardrail import SafetyAndRelevanceGuardrail
from src.llm.llm_client import FastLLMClient
from src.retrieval.retriever import DenseRetriever


class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    guardrail_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


class RAGResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    latency: LatencyBreakdown
    status: str = "success"


class PipelineHarness:

    def __init__(
        self,
        retriever: DenseRetriever,
        llm_client: FastLLMClient,
        guardrail: SafetyAndRelevanceGuardrail,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.guardrail = guardrail

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.1, min=0.1, max=0.5)
    )
    def _call_llm_with_retry(
        self, query: str, contexts: List[Dict[str, Any]]
    ) -> str:
        return self.llm_client.generate_answer(query, contexts)

    def execute_pipeline(
        self, query: str, stt_latency_ms: float = 0.0, top_k: int = 3
    ) -> RAGResponse:
        total_start = time.perf_counter()
        breakdown = LatencyBreakdown(stt_ms=stt_latency_ms)

        # 1. Input Guardrail
        g_start = time.perf_counter()
        is_safe, safety_msg = self.guardrail.check_input_safety(query)
        breakdown.guardrail_ms += (time.perf_counter() - g_start) * 1000

        if not is_safe:
            total_duration = (time.perf_counter() - total_start) * 1000
            breakdown.total_ms = round(total_duration + stt_latency_ms, 2)
            return RAGResponse(
                query=query,
                answer=safety_msg,
                grounded=False,
                latency=breakdown,
                status="rejected_by_guardrail",
            )

        # 2. Retrieval & Embedding
        r_start = time.perf_counter()
        retrieved_results = self.retriever.retrieve(query, top_k=top_k)
        breakdown.retrieval_ms = round(
            (time.perf_counter() - r_start) * 1000, 2
        )

        # 3. Context Grounding Guardrail
        g_start2 = time.perf_counter()
        is_grounded, grounding_msg = (
            self.guardrail.check_grounding_and_relevance(retrieved_results)
        )
        breakdown.guardrail_ms += (time.perf_counter() - g_start2) * 1000
        breakdown.guardrail_ms = round(breakdown.guardrail_ms, 2)

        if not is_grounded:
            total_duration = (time.perf_counter() - total_start) * 1000
            breakdown.total_ms = round(total_duration + stt_latency_ms, 2)
            return RAGResponse(
                query=query,
                answer=grounding_msg,
                grounded=False,
                latency=breakdown,
                status="refused_low_confidence",
            )

        # 4. LLM Generation
        matched_chunks = [item[0] for item in retrieved_results]
        l_start = time.perf_counter()
        answer = self._call_llm_with_retry(query, matched_chunks)
        breakdown.llm_ms = round((time.perf_counter() - l_start) * 1000, 2)

        total_duration = (time.perf_counter() - total_start) * 1000
        breakdown.total_ms = round(total_duration + stt_latency_ms, 2)

        return RAGResponse(
            query=query,
            answer=answer,
            grounded=True,
            sources=matched_chunks,
            latency=breakdown,
            status="success",
        )