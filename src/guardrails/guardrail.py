import re
from typing import Any, Dict, List


class SafetyAndRelevanceGuardrail:

    def __init__(self, similarity_threshold: float = 0.0):
        self.similarity_threshold = similarity_threshold
        self.unsafe_patterns = [
            r"ignore previous instructions",
            r"bypass system",
            r"system prompt",
            r"drop database",
            r"jailbreak",
        ]

    def is_safe_query(self, query: str) -> bool:
        """Checks query against injection patterns and unsafe manipulation."""
        if not query or not str(query).strip():
            return False
        q_lower = str(query).lower()
        for pattern in self.unsafe_patterns:
            if re.search(pattern, q_lower):
                return False
        return True

    def validate_query(self, query: str) -> bool:
        return self.is_safe_query(query)

    def is_relevant_context(
        self, query: str, retrieved_chunks: List[Any]
    ) -> bool:
        """Verifies if the retrieved context is present and passes threshold."""
        if not retrieved_chunks:
            return False

        first = retrieved_chunks[0]
        if isinstance(first, dict):
            top_score = float(first.get("score", 1.0))
        elif isinstance(first, tuple) and len(first) >= 2:
            top_score = float(first[1])
        else:
            top_score = 1.0

        return top_score >= self.similarity_threshold

    def validate_response(self, response: str) -> bool:
        return bool(response and len(str(response).strip()) > 5)