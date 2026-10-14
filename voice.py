"""Voice in / voice out with OpenAI mini audio models."""

import io
import re

from openai import OpenAI

from config import STT_MODEL, TTS_MODEL, TTS_VOICE

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def transcribe(audio_bytes: bytes, filename: str = "question.wav") -> str:
    """Speech -> text. Language is auto-detected (Tamil or English)."""
    f = io.BytesIO(audio_bytes)
    f.name = filename  # the extension tells OpenAI the audio format
    return client().audio.transcriptions.create(model=STT_MODEL, file=f).text.strip()


def to_speech_text(markdown: str, limit: int = 1500) -> str:
    """Strip markdown symbols so the voice reads naturally; cap length for latency."""
    text = re.sub(r"[#*_`>|]+", " ", markdown)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def speak(markdown: str, language: str) -> bytes:
    """Text -> MP3 bytes in the given language."""
    lang_name = "Tamil" if language == "ta" else "Indian English"
    resp = client().audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=to_speech_text(markdown),
        # "Read verbatim" stops the TTS model from occasionally replying to the text instead of reading it.
        instructions=f"Read the text exactly as written, word for word, in {lang_name}. Do not add, answer or "
                     "change anything. Use a warm, slow, clear voice, like an agriculture officer talking to a farmer.",
        response_format="mp3",
    )
    return resp.content
