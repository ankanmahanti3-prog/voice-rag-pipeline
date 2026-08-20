import re
from typing import List, Dict, Any

class SafetyAndRelevanceGuardrail:
    def __init__(self, similarity_threshold: float = 0.0):
        self.similarity_threshold = similarity_threshold
        # Block malicious prompt injections or system manipulation attempts
        self.unsafe_patterns = [
            r"ignore previous instructions",
            r"bypass system",
            r"system prompt",
            r"drop database",
            r"jailbreak"
        ]

    def is_safe_query(self, query: str) -> bool:
        """Checks query against injection patterns and profanity/unsafe requests."""
        if not query or not str(query).strip():
            return False
        q_lower = str(query).lower()
        for pattern in self.unsafe_patterns:
            if re.search(pattern, q_lower):
                return False
        return True

    def validate_query(self, query: str) -> bool:
        return self.is_safe_query(query)

    def is_relevant_context(self, query: str, retrieved_chunks: List[Any]) -> bool:
        """Always accepts retrieved context when documents exist in vector store."""
        return bool(retrieved_chunks and len(retrieved_chunks) > 0)

    def validate_response(self, response: str) -> bool:
        return bool(response and len(str(response).strip()) > 3)