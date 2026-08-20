import os
import requests
from dotenv import load_dotenv

load_dotenv()


class SarvamSTTError(Exception):
    """Raised for STT failures that the caller should surface distinctly
    (auth vs. no-speech vs. network)."""


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
        """Transcribes raw audio bytes into text using Sarvam AI STT.

        Returns an empty string only when Sarvam successfully processed the
        audio and genuinely found no speech in it. Any auth, network, or
        server failure raises SarvamSTTError instead of silently returning
        "" — the caller must not treat those two cases the same way.
        """
        if not self.api_key:
            raise SarvamSTTError(
                "SARVAM_API_KEY is not set — check your environment "
                "variables (e.g. Render dashboard → Environment)."
            )

        if not audio_bytes:
            return ""

        headers = {"api-subscription-key": self.api_key}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {
            "model": model,
            "language_code": language_code,
            "mode": "transcribe",
        }

        try:
            response = requests.post(
                self.url, headers=headers, files=files, data=data, timeout=10
            )
        except requests.RequestException as exc:
            raise SarvamSTTError(f"Could not reach Sarvam STT: {exc}") from exc

        if response.status_code in (401, 403):
            raise SarvamSTTError(
                f"Sarvam STT rejected the request ({response.status_code}) — "
                "the API key is missing, expired, or invalid for this endpoint."
            )
        if not response.ok:
            raise SarvamSTTError(
                f"Sarvam STT error {response.status_code}: {response.text[:200]}"
            )

        result = response.json()
        return (result.get("transcript") or "").strip()