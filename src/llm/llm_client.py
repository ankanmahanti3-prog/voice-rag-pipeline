import os
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()


class FastLLMClient:

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.client = None
        self.active_model = "llama-3.1-8b-instant"
        self._cache = {}

        try:
            from groq import Groq

            if self.api_key:
                self.client = Groq(api_key=self.api_key, timeout=0.18)
        except Exception as e:
            print(f"Groq Init Notice: {e}")

    def generate_answer(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 25,
    ) -> str:
        """Fast grounded answer with local response caching and strict latency bounding."""
        clean_query = query.strip().lower()
        if clean_query in self._cache:
            return self._cache[clean_query]

        clean_passages = []
        for doc in retrieved_contexts:
            text = doc.get("text", "")
            if "\n" in text and text.startswith("[Context:"):
                text = text.split("\n", 1)[1]
            clean_passages.append(text.strip())

        context_snippet = (
            clean_passages[0]
            if clean_passages
            else "No matching context found."
        )

        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer directly in 1 short sentence using only the context.",
                        },
                        {
                            "role": "user",
                            "content": f"Context: {context_snippet}\nQuestion: {query}",
                        },
                    ],
                    model=self.active_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                ans = completion.choices[0].message.content.strip()
                if ans:
                    self._cache[clean_query] = ans
                    return ans
            except Exception:
                pass

        ans = (
            context_snippet
            if clean_passages
            else "I do not have sufficient information in the knowledge base."
        )
        self._cache[clean_query] = ans
        return ans