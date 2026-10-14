"""Shared settings for ingest.py and app.py. Override any value via .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "source-data"
INDEX_DIR = ROOT / "faiss_index"
CATALOG_FILE = ROOT / "schemes_catalog.json"

API_URL = os.getenv("API_URL", "http://localhost:8000")  # FastAPI backend used by app.py

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "coral")

# Friendly source names shown as citations in the chat.
SOURCE_TITLES = {
    "agri_e_pn_2026_27.pdf": "TN Agriculture Policy Note 2026-27",
    "BookIIISchemesVolIF.pdf": "ICAR-SBI: TN Agri Schemes Made Simple, Vol I (Central)",
    "BookIIISchemesVolIIF.pdf": "ICAR-SBI: TN Agri Schemes Made Simple, Vol II (State)",
    "tnagrisnet_schemes.md": "TNAGRISNET scheme portal",
    "tnagrisnet_goscheme_subsidies.md": "TNAGRISNET GO & subsidy list",
    "tn_gov_agri_schemes.md": "tn.gov.in scheme portal",
}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 6
