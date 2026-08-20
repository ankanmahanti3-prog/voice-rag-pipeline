import os
import time
from typing import Optional, Tuple
import requests


class SarvamSTTClient:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("SARVAM_API_KEY")
            or "sk_r2vbg3c2_bu2jW8sHdpwa9k35FcR4HkCY"
        )
        self.url = "https://api.sarvam.ai/speech-to-text"

    def transcribe(
        self, audio_bytes: bytes, language_code: str = "en-IN"
    ) -> Tuple[str, float]:
        """Transcribes audio using Sarvam Saaras with English / Indic auto-transcription."""
        start_time = time.perf_counter()

        if not self.api_key:
            return "What is RAG?", 1.0

        headers = {"api-subscription-key": self.api_key}
        files = {
            "file": (
                "audio.wav",
                audio_bytes,
                "audio/wav",
            )  # Explicit MIME type for fast ingestion
        }
        data = {
            "model": "saaras:v3",
            "language_code": language_code,  # 'en-IN' ensures English transcriptions
        }

        try:
            response = requests.post(
                self.url, headers=headers, files=files, data=data, timeout=5.0
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                transcript = result.get("transcript", "").strip()
                return transcript, elapsed_ms
            else:
                print(f"STT Error {response.status_code}: {response.text}")
                return "", elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"STT Exception: {e}")
            return "", elapsed_ms