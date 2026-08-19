import os
import requests
from dotenv import load_dotenv

load_dotenv()


class SarvamSTTClient:

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.url = "https://api.sarvam.ai/speech-to-text"

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: str = "hi-IN",
        model: str = "saaras:v3",
    ) -> str:
        """Transcribes raw audio bytes into text using Sarvam AI STT."""
        headers = {"api-subscription-key": self.api_key}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {
            "model": model,
            "language_code": language_code,
            "mode": "transcribe",
        }

        response = requests.post(
            self.url, headers=headers, files=files, data=data, timeout=10
        )
        response.raise_for_status()
        result = response.json()
        return result.get("transcript", "")