"""UzhavarGPT backend: FastAPI wraps the RAG pipeline and voice; Streamlit (app.py) is the UI.

    uvicorn api:app --port 8000        # interactive docs at http://localhost:8000/docs

Endpoints
  GET  /health      index + key status
  GET  /schemes     scheme directory (schemes_catalog.json)
  POST /chat        streamed answer as NDJSON lines:
                      {"type": "meta", "language": "ta", "sources": [...]}
                      {"type": "token", "text": "..."}   (repeated)
  POST /transcribe  audio file -> {"text": ...}
  POST /speak       {"text", "language"} -> audio/mpeg
"""

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

import voice
from config import CATALOG_FILE, INDEX_DIR
from rag import SchemeRAG, detect_language, format_sources

state = {"rag": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load FAISS once at startup (read-only). Missing index/key -> /health reports it.
    if (INDEX_DIR / "index.faiss").exists() and os.getenv("OPENAI_API_KEY"):
        state["rag"] = SchemeRAG()
    yield


app = FastAPI(title="UzhavarGPT API", version="1.0", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


class SpeakRequest(BaseModel):
    text: str
    language: str = "en"


def require_rag() -> SchemeRAG:
    if state["rag"] is None:
        raise HTTPException(503, "Knowledge base not loaded. Set OPENAI_API_KEY, run `python ingest.py`, restart the API.")
    return state["rag"]


@app.get("/health")
def health():
    manifest = INDEX_DIR / "manifest.json"
    return {
        "status": "ok",
        "index_ready": state["rag"] is not None,
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
    }


@app.get("/schemes")
def schemes():
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


@app.post("/chat")
def chat(req: ChatRequest):
    rag = require_rag()
    history = [m.model_dump() for m in req.history]
    language = detect_language(req.question)
    _, docs = rag.retrieve(req.question, history)

    def events():
        yield json.dumps({"type": "meta", "language": language, "sources": format_sources(docs)}, ensure_ascii=False) + "\n"
        for token in rag.stream_answer(req.question, history, docs, language):
            yield json.dumps({"type": "token", "text": token}, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    require_rag()
    return {"text": voice.transcribe(audio.file.read(), audio.filename or "question.wav")}


@app.post("/speak")
def speak(req: SpeakRequest):
    require_rag()
    return Response(voice.speak(req.text, req.language), media_type="audio/mpeg")
