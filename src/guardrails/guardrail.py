from typing import Any, Dict, List, Tuple
import re


class SafetyAndRelevanceGuardrail:

    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold
        # Block malicious/jailbreak prompt injections
        self.banned_patterns = [
            r"ignore all previous instructions",
            r"system override",
            r"drop table",
            r"act as an unfiltered",
        ]

    def check_input_safety(self, query: str) -> Tuple[bool, str]:
        """Validates query against malicious patterns and jailbreak attempts."""
        if not query or not query.strip():
            return False, "Query cannot be empty."

        query_lower = query.lower()
        for pattern in self.banned_patterns:
            if re.search(pattern, query_lower):
                return False, "Query rejected by security guardrail."

        return True, "Passed"

    def check_grounding_and_relevance(
        self, retrieved_results: List[Tuple[Dict[str, Any], float]]
    ) -> Tuple[bool, str]:
        """Validates if retrieved context is sufficiently relevant to answer grounded in facts."""
        if not retrieved_results:
            return (
                False,
                "I do not have sufficient context from the dataset to answer this question.",
            )

        top_score = retrieved_results[0][1]
        if top_score < self.similarity_threshold:
            return (
                False,
                f"Low context relevance confidence ({top_score:.2f}). Refusing answer to prevent hallucination.",
            )

        return True, "Passed"


if __name__ == "__main__":
    guard = SafetyAndRelevanceGuardrail(similarity_threshold=0.4)
    safe, msg = guard.check_input_safety("Ignore all previous instructions")
    print(f"Safety Check: {safe} -> {msg}")