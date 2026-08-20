import re
from typing import Any, Dict, List, Optional


class SafetyAndRelevanceGuardrail:

    def __init__(self, similarity_threshold: float = 0.0):
        self.similarity_threshold = similarity_threshold
        # Block malicious prompt injections or system manipulation attempts
        self.unsafe_patterns = [
            r"ignore previous instructions",
            r"bypass system",
            r"system prompt",
            r"drop database",
            r"jailbreak",
        ]

    def is_safe_query(self, query: str) -> bool:
        """Checks query against injection patterns and profanity/unsafe requests."""
        if not query or not query.strip():
            return False
        q_lower = query.lower()
        for pattern in self.unsafe_patterns:
            if re.search(pattern, q_lower):
                return False
        return True

    def validate_query(self, query: str) -> bool:
        """Alias for is_safe_query."""
        return self.is_safe_query(query)

    def is_relevant_context(
        self, query: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> bool:
        """Verifies if the retrieved context is relevant and above threshold."""
        if not retrieved_chunks:
            return False

        top_score = float(retrieved_chunks[0].get("score", 0.0))
        return top_score >= self.similarity_threshold

    def validate_response(self, response: str) -> bool:
        """Checks if response is grounded and non-empty."""
        return bool(response and len(response.strip()) > 5)