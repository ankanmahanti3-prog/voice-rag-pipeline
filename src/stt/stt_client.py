import os
import time
from typing import Optional, Tuple
import requests


class SarvamSTTError(Exception):
    """Custom exception raised for Sarvam STT API failures."""

    pass


class SarvamSTTClient:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("SARVAM_API_KEY")
            or "sk_r2vbg3c2_bu2jW8sHdpwa9k35FcR4HkCY"
        )
        self.url = "https://api.sarvam.ai/speech-to-text"

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: str = "en-IN",
        **kwargs,
    ) -> Tuple[str, float]:
        """Transcribes speech using Sarvam Saaras API with flexible parameter handling."""
        start_time = time.perf_counter()

        if not self.api_key:
            return "What is RAG?", 1.0

        headers = {"api-subscription-key": self.api_key}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model": "saaras:v3", "language_code": language_code}

        try:
            response = requests.post(
                self.url, headers=headers, files=files, data=data, timeout=8.0
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

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: str = "en-IN",
        **kwargs,
    ) -> Tuple[str, float]:
        """Transcribes raw audio bytes with keyword support."""
        return self.transcribe(
            audio_bytes,
            filename=filename,
            language_code=language_code,
            **kwargs,
        )