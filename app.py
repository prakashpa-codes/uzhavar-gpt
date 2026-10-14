"""UzhavarGPT - bilingual (English / Tamil) RAG assistant for TN agriculture schemes.

UI only: all data, answers and voice come from the FastAPI backend (api.py).

    uvicorn api:app --port 8000
    streamlit run app.py
"""

import html
import json
import time
import uuid

import requests
import streamlit as st

from config import API_URL
from ui_art import loader_html, paddy_field

st.set_page_config(page_title="UzhavarGPT | உழவர்GPT", page_icon="🌾", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ UI text
T = {
    "en": {
        "app_name": "UzhavarGPT", "app_alt": "உழவர்GPT",
        "tagline": "Tamil Nadu & Central agriculture schemes, made simple",
        "new_chat": "New chat", "clear": "Clear", "history": "History", "font": "Text size", "sizes": {"s": "Small", "m": "Normal", "l": "Large", "xl": "Larger"},
        "no_history": "No conversations yet", "untitled": "New conversation",
        "directory": "Scheme Directory", "directory_sub": "{n} schemes · click any scheme to ask about it",
        "search": "Search schemes…", "all": "All", "state": "State", "central": "Central",
        "recent": "Recent", "popular": "Popular", "ask": "Ask ➜", "quick": "Quick Questions",
        "chat_title": "Ask your agriculture questions",
        "chat_sub": "Answers come only from official scheme documents",
        "welcome": "**Vanakkam! 🙏** I can help you with Tamil Nadu and Central government agriculture "
                   "schemes: subsidies, eligibility, documents and how to apply.\n\n"
                   "Ask in **English or Tamil**, type below or 🎤 speak your question.",
        "placeholder": "Ask about schemes, subsidies, eligibility… (English / தமிழ்)",
        "voice_label": "🎤 Or tap the mic and speak in Tamil or English. You'll hear the answer too.",
        "transcribing": "Listening…", "searching": "Searching scheme documents…",
        "speaking": "Preparing voice reply…", "sources": "Sources", "voice_note": "Voice question",
        "no_match": "No schemes match your filters.",
        "scheme_q": "Explain the {name} scheme: benefits, eligibility, documents required and how to apply.",
        "no_api": "Backend not reachable at {url}. Start it with `uvicorn api:app --port 8000`.",
        "no_index": "Knowledge base not loaded. Add `OPENAI_API_KEY` to `.env`, run `python ingest.py`, then restart the API.",
        "error": "Sorry, something went wrong: {e}",
        "disclaimer": "Note: Please verify schemes, eligibility and documents with official government "
                      "sources or your nearest Agricultural Extension Centre.",
        "kural_meaning": "The world follows the plough; so farming, though hard, is the foremost occupation.",
        "lvl": {"Central": "Central", "State": "State", "Central + State": "Central + State"},
    },
    "ta": {
        "app_name": "உழவர்GPT", "app_alt": "UzhavarGPT",
        "tagline": "வேளாண் நலத் திட்டங்கள், எளிமையாக",
        "new_chat": "புதிய உரையாடல்", "clear": "அழி", "history": "உரையாடல்கள்", "font": "எழுத்து அளவு", "sizes": {"s": "சிறியது", "m": "இயல்பு", "l": "பெரியது", "xl": "மிகப் பெரியது"},
        "no_history": "இதுவரை உரையாடல்கள் இல்லை", "untitled": "புதிய உரையாடல்",
        "directory": "திட்ட அடைவு", "directory_sub": "{n} திட்டங்கள் · ஏதேனும் ஒரு திட்டத்தைச் சொடுக்கிக் கேளுங்கள்",
        "search": "திட்டங்களைத் தேடுக…", "all": "அனைத்தும்", "state": "மாநில அரசு", "central": "மத்திய அரசு",
        "recent": "புதியவை", "popular": "பிரபலமானவை", "ask": "கேள் ➜", "quick": "விரைவான கேள்விகள்",
        "chat_title": "உங்கள் விவசாயக் கேள்விகளைக் கேளுங்கள்",
        "chat_sub": "பதில்கள் அதிகாரப்பூர்வ திட்ட ஆவணங்களிலிருந்து மட்டுமே",
        "welcome": "**வணக்கம்! 🙏** தமிழ்நாடு மற்றும் மத்திய அரசின் வேளாண் திட்டங்கள், மானியம், தகுதி, "
                   "தேவையான ஆவணங்கள், விண்ணப்பிக்கும் முறை குறித்து உதவ நான் தயார்.\n\n"
                   "**தமிழ் அல்லது ஆங்கிலத்தில்** கீழே தட்டச்சு செய்யுங்கள் அல்லது 🎤 பேசுங்கள்.",
        "placeholder": "திட்டங்கள், மானியம், தகுதி பற்றிக் கேளுங்கள்… (தமிழ் / English)",
        "voice_label": "🎤 அல்லது மைக்கைத் தொட்டு தமிழ் / ஆங்கிலத்தில் பேசுங்கள். பதிலைக் கேட்கவும் முடியும்.",
        "transcribing": "கேட்கிறேன்…", "searching": "திட்ட ஆவணங்களில் தேடுகிறேன்…",
        "speaking": "குரல் பதிலைத் தயாரிக்கிறேன்…", "sources": "ஆதாரங்கள்", "voice_note": "குரல் கேள்வி",
        "no_match": "உங்கள் தேடலுக்குப் பொருந்தும் திட்டங்கள் இல்லை.",
        "scheme_q": "{name} திட்டத்தின் பயன்கள், தகுதி, தேவையான ஆவணங்கள் மற்றும் விண்ணப்பிக்கும் முறையை விளக்குங்கள்.",
        "no_api": "{url} இல் சேவையகம் (backend) இயங்கவில்லை. `uvicorn api:app --port 8000` மூலம் தொடங்கவும்.",
        "no_index": "அறிவுத் தளம் ஏற்றப்படவில்லை. `.env` இல் `OPENAI_API_KEY` சேர்த்து, `python ingest.py` இயக்கி, API ஐ மீண்டும் தொடங்கவும்.",
        "error": "மன்னிக்கவும், பிழை ஏற்பட்டது: {e}",
        "disclaimer": "குறிப்பு: திட்டங்கள், தகுதி மற்றும் ஆவணங்களை அதிகாரப்பூர்வ அரசு ஆதாரங்கள் அல்லது "
                      "அருகிலுள்ள வேளாண் விரிவாக்க மையத்தில் சரிபார்க்கவும்.",
        "kural_meaning": "உலகம் ஏரின் பின்னே செல்கிறது; ஆகவே உழவே தலைசிறந்த தொழில்.",
        "lvl": {"Central": "மத்திய அரசு", "State": "மாநில அரசு", "Central + State": "மத்திய + மாநில"},
    },
}
KURAL = "சுழன்றும்ஏர்ப் பின்னது உலகம் அதனால்<br>உழந்தும் உழவே தலை."


# ------------------------------------------------------------------ backend API client
def api_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_catalog() -> dict:
    resp = requests.get(f"{API_URL}/schemes", timeout=10)
    resp.raise_for_status()
    return resp.json()


def stream_chat(question: str, history: list[dict], meta: dict):
    """Yield answer tokens from POST /chat; the first NDJSON line (sources, language) goes into `meta`."""
    payload = {"question": question, "history": [{"role": m["role"], "content": m["content"]} for m in history]}
    with requests.post(f"{API_URL}/chat", json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "meta":
                meta.update(event)
            else:
                yield event["text"]


def api_transcribe(audio_bytes: bytes) -> str:
    resp = requests.post(f"{API_URL}/transcribe", files={"audio": ("question.wav", audio_bytes, "audio/wav")}, timeout=60)
    resp.raise_for_status()
    return resp.json()["text"]


def api_speak(text: str, language: str) -> bytes:
    resp = requests.post(f"{API_URL}/speak", json={"text": text, "language": language}, timeout=60)
    resp.raise_for_status()
    return resp.content


# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("chats", {})
ss.setdefault("chat_id", None)
ss.setdefault("pending", None)
ss.setdefault("font_scale", 1.0)
ss.setdefault("audio_key", 0)
lang = ss.get("lang_ctl") or ss.get("lang", "ta")  # Tamil by default; keep last choice if the toggle is deselected
ss.lang = lang
t = T[lang]
health = api_health()
if health is None:
    st.error(t["no_api"].format(url=API_URL))
    st.stop()
catalog = load_catalog()
ready = health["index_ready"]


def new_chat():
    current = ss.chats.get(ss.chat_id)
    if current is not None and not current["messages"]:
        return  # already on an empty chat
    cid = uuid.uuid4().hex[:8]
    ss.chats[cid] = {"title": None, "messages": [], "created": time.strftime("%d %b, %H:%M")}
    ss.chat_id = cid


def clear_chat():
    ss.chats[ss.chat_id].update(title=None, messages=[])


def switch_chat(cid):
    ss.chat_id = cid


FONT_SIZES = {"s": 0.9, "m": 1.0, "l": 1.15, "xl": 1.3}


def set_font():
    ss.font_scale = FONT_SIZES[ss.font_ctl or "m"]


def ask(question):
    ss.pending = question


if ss.chat_id is None:
    new_chat()
chat = ss.chats[ss.chat_id]

# ------------------------------------------------------------------ styles
fs = ss.font_scale
ui = fs * 0.9  # base UI text one step smaller; scheme cards stay at fs
field_bg = paddy_field(1400, 150, 230, color="#b88a3b", grain="#8b5a2b", alpha=0.3)
header_art = paddy_field(560, 90, 70, color="#ffffff", grain="#e2b23a", alpha=0.14, seed=3)
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap');
:root {{
  --g900:#4a2716; --g700:#6b3a22; --g600:#a0522d; --g200:#e8d3bf; --g100:#f5e9dc; --g50:#fbf5ee;
  --cream:#fbf6e7; --gold:#b98b12; --leaf:#4e7d3a;
  --h900:#0f3d1f; --h700:#1b6b34; --h500:#2f8a44;  /* green header + titles */ --ink:#2b1d14; --muted:#7a6656;
}}
.stApp {{
  background:
    url("{field_bg}") bottom center / 1400px 150px repeat-x fixed,
    radial-gradient(900px 480px at 88% -8%, rgba(226,178,58,.28), transparent 62%) fixed,
    radial-gradient(760px 620px at -8% 35%, rgba(78,125,58,.10), transparent 60%) fixed,
    linear-gradient(180deg, #f6eee3 0%, #fbf7f0 45%, #f3e9da 100%) fixed;
}}
/* no "stale" fade while the app reruns: the paddy loader shows progress instead. (Partial opacity would also
   create stacking contexts that trap the fixed loader underneath later elements.) */
[data-stale="true"] {{ opacity: 1 !important; }}
.stApp, .stApp p, .stApp li, .stApp input, .stApp textarea, .stApp button p, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  font-family: 'Inter', 'Noto Sans Tamil', sans-serif; color: var(--ink);
}}
[data-testid="stHeader"], #MainMenu, footer {{ display: none; }}
[data-testid="stMainBlockContainer"] {{ padding: 0.9rem 1.6rem 0.5rem; max-width: 1560px; }}

/* font scaling */
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp textarea, .stApp input {{ font-size: {15 * ui:.1f}px; }}
.stApp button p {{ font-size: {13.5 * ui:.1f}px; }}
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2 {{ font-size: {19 * ui:.1f}px; }}
[data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 {{ font-size: {16.5 * ui:.1f}px; padding: .4rem 0 .1rem; }}

/* top bar */
.st-key-topbar {{
  background: url("{header_art}") right bottom / 560px 100% no-repeat,
              radial-gradient(420px 120px at 72% 0%, rgba(226,178,58,.22), transparent 70%),
              linear-gradient(100deg, var(--h900), var(--h700) 60%, var(--h500));
  border-radius: 18px; padding: 10px 18px;
  box-shadow: 0 6px 20px rgba(15,61,31,.2), inset 0 -2px 0 rgba(226,178,58,.55);
}}
.st-key-topbar p, .st-key-topbar span {{ color: #fff !important; }}
.st-key-topbar button {{
  background: transparent !important; border: 1px solid transparent !important; border-radius: 10px !important;
  box-shadow: none !important;
}}
.st-key-topbar button:hover {{ background: rgba(255,255,255,.14) !important; }}
.st-key-topbar [data-testid="stIconMaterial"] {{ color: #fff !important; font-size: 1.35rem; }}
.st-key-topbar button p {{ font-weight: 500; }}
/* language: one quiet pill, both options always visible (easy for first-time users) */
.st-key-lang_ctl [role="radiogroup"] {{
  column-gap: 2px; padding: 3px; width: fit-content; margin-left: auto; border-radius: 999px;
  background: rgba(0,0,0,.18); border: 1px solid rgba(255,255,255,.22);
}}
.st-key-lang_ctl button {{ border-radius: 999px !important; }}
.st-key-topbar .st-key-lang_ctl button[aria-checked="true"] {{ background: rgba(255,255,255,.95) !important; }}
.st-key-topbar .st-key-lang_ctl button[aria-checked="true"] p {{ color: var(--h900) !important; font-weight: 700; }}
/* new chat: the one emphasised action, top-right corner */
.st-key-new_chat_btn button {{ background: rgba(255,255,255,.14) !important; border-color: rgba(255,255,255,.28) !important; }}
.st-key-new_chat_btn button:hover {{ background: rgba(255,255,255,.26) !important; }}
.st-key-topbar [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.brand .logo {{ font-size: 2.1rem; background: rgba(255,255,255,.15); border-radius: 50%;
  width: 52px; height: 52px; display:flex; align-items:center; justify-content:center; }}
.brand .name {{ font-size: 1.45rem; font-weight: 700; color:#fff; line-height:1.1; }}
.brand .alt {{ font-size: .95rem; font-weight: 500; color:#d6ecd0; margin-left: 6px; }}
.brand .tag {{ font-size: .8rem; color:#d6ecd0; }}

/* panels */
.st-key-catalog, .st-key-chatpane {{
  background: rgba(255,255,255,.88); border: 1px solid var(--g200); border-radius: 18px;
  padding: 16px 18px; box-shadow: 0 4px 16px rgba(107,58,34,.07);
}}
.st-key-chatpane {{ border: 2px solid var(--g600); }}
.panel-title {{ font-size: {20 * ui:.1f}px; font-weight: 700; color: var(--h900); margin: 0; }}
.panel-sub {{ font-size: {13 * ui:.1f}px; color: var(--muted); margin-top: 2px; }}

/* scheme cards */
[class*="st-key-card_"] {{ background: var(--g50); border-radius: 14px !important; border-color: var(--g200) !important; }}
[class*="st-key-card_"]:hover {{ border-color: var(--g600) !important; box-shadow: 0 3px 12px rgba(107,58,34,.12); }}
.card {{ font-size: {14 * fs:.1f}px; }}
.card .top {{ display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom: 4px; }}
.card .badge {{ font-size: .72em; font-weight:600; border-radius: 999px; padding: 2px 8px; }}
.badge.central {{ background:#e3edfb; color:#1d4f91; }}
.badge.state {{ background:#e6f4e1; color:#3f6b2e; }}
.badge.both {{ background:#efe8fb; color:#5b3a99; }}
.badge.recent {{ background:#fff1d6; color:#9a6200; }}
.badge.popular {{ background:#ffe3e0; color:#b23a2a; }}
.card .nm {{ font-weight: 650; color: var(--h900); line-height: 1.3; }}
.card .alt {{ font-size: .8em; color: var(--muted); margin-top: 1px; }}
.card .ds {{ font-size: .86em; color: #4a3a2e; margin-top: 6px; line-height: 1.4; }}
[class*="st-key-card_"] button {{ border-radius: 999px; border-color: var(--g600); }}
[class*="st-key-card_"] button p {{ color: var(--g700); font-weight: 600; }}

/* quick questions */
.st-key-quickq button {{ justify-content: flex-start; text-align: left; border-radius: 999px;
  background: #fff; border: 1px solid var(--g200); }}
.st-key-quickq button:hover {{ border-color: var(--g600); background: var(--g50); }}
.section-h {{ font-size: {16 * ui:.1f}px; font-weight: 700; color: var(--h900); margin: 0 0 4px; }}
/* defined sections inside the directory panel */
.st-key-cardgrid {{ border: 1px solid var(--g200); border-radius: 14px; background: rgba(251,245,238,.6);
  padding: 10px 10px 4px; box-shadow: inset 0 -10px 12px -12px rgba(74,39,22,.25); }}
.st-key-quickq {{ border: 1px solid var(--g200); border-radius: 14px; background: #fff; padding: 12px 14px 14px; }}

/* heritage */
.heritage {{ background: var(--cream); border: 1px solid #ecdcae; border-radius: 14px; padding: 12px 16px; text-align:center;
  margin-bottom: 20px; }}
.heritage .k {{ font-family:'Noto Sans Tamil',sans-serif; font-weight:600; color:#3d2e00; font-size: {15 * ui:.1f}px; margin: 0 0 4px; }}
.heritage .m {{ font-size: {12.5 * ui:.1f}px; color:#6b5a2e; }}

/* chat */
[data-testid="stChatMessage"] {{ background: #fff; border: 1px solid #efe2d3; border-radius: 14px; padding: .7rem .9rem; }}
.src {{ font-size: {12 * ui:.1f}px; color: var(--muted); border-top: 1px dashed var(--g200); padding-top: 6px; margin-top: 4px; }}
/* chat input: the primary action, so make it obvious */
[data-testid="stChatInput"] {{ border: none !important; border-radius: 16px; }}
[data-testid="stChatInput"] > div {{
  background: #fff !important; border: 2px solid var(--g600) !important; border-radius: 16px !important;
  box-shadow: 0 4px 16px rgba(107,58,34,.14);
}}
[data-testid="stChatInput"] > div:focus-within {{ border-color: var(--g700) !important; box-shadow: 0 0 0 4px rgba(226,178,58,.45), 0 4px 16px rgba(107,58,34,.14); }}
[data-testid="stChatInput"] textarea {{ font-size: {15.5 * ui:.1f}px; }}
[data-testid="stChatInputSubmitButton"] {{ background: var(--g600) !important; border-radius: 12px !important; }}
[data-testid="stChatInputSubmitButton"] svg {{ color: #fff !important; fill: #fff !important; }}
[data-testid="stChatInputSubmitButton"]:disabled {{ background: var(--g200) !important; }}
/* voice: secondary, subtle */
.voice-hint {{ font-size: {12.5 * ui:.1f}px; color: var(--muted); margin: 4px 2px 6px; }}
[data-testid="stAudioInput"] > div {{ background: var(--g50) !important; border: 1px dashed var(--g200); border-radius: 12px; }}
.disclaimer {{ text-align:center; font-size: 12px; color:#f2efe2; background: #4a2716; border-radius: 12px;
  padding: 8px 14px; margin-top: 12px; }}
/* desktop: chat area fits the screen so the input + mic are visible on first load */
[data-testid="stLayoutWrapper"]:has(> .st-key-chatbox) {{
  height: clamp(260px, calc(100vh - 470px), 640px) !important; flex: 0 0 auto !important;
}}
/* mobile: compact header, chat first, extras hidden */
@media (max-width: 768px) {{
  [data-testid="stMainBlockContainer"] {{ padding: .5rem .6rem 0; }}
  .st-key-topbar {{ padding: 8px 10px; background: linear-gradient(100deg, var(--h900), var(--h700)) !important; }}
  .st-key-topbar [data-testid="stHorizontalBlock"] {{ flex-flow: row wrap !important; gap: 6px !important; }}
  .st-key-topbar [data-testid="stColumn"] {{ min-width: 0 !important; width: auto !important; flex: 1 1 auto !important; }}
  .st-key-topbar [data-testid="stColumn"]:first-child {{ flex-basis: 100% !important; }}
  .brand .tag {{ display: none; }}
  .brand .logo {{ width: 38px; height: 38px; font-size: 1.5rem; }}
  .brand .name {{ font-size: 1.2rem; }}
  [data-testid="stHorizontalBlock"]:has(.st-key-chatpane) {{ flex-direction: column-reverse !important; }}
  .st-key-catalog, .st-key-chatpane {{ padding: 12px; }}
  .st-key-quickq, .heritage {{ display: none; }}
  .st-key-cat_ctl div:has(> button) {{ flex-wrap: nowrap !important; overflow-x: auto; scrollbar-width: none; padding-bottom: 2px; }}
  .st-key-cat_ctl button {{ flex-shrink: 0; }}
  /* fixed-height containers: Streamlit puts the height on the wrapper around the keyed block */
  [data-testid="stLayoutWrapper"]:has(> .st-key-chatbox) {{ height: 340px !important; flex: 0 0 340px !important; }}
  [data-testid="stLayoutWrapper"]:has(> .st-key-cardgrid) {{ height: 380px !important; flex: 0 0 380px !important; }}
  .disclaimer {{ font-size: 11px; }}
}}
/* paddy loader: centered; stalks rise one by one, left to right. Visible only while Streamlit runs. */
.paddy-loader {{
  position: fixed; left: 50%; top: 50%; z-index: 999999; pointer-events: none;
  padding: 14px 20px 10px; border-radius: 20px; background: rgba(255,255,255,.9); backdrop-filter: blur(8px);
  border: 1px solid var(--g200); box-shadow: 0 12px 32px rgba(74,39,22,.18);
  opacity: 0; transform: translate(-50%, -50%) scale(.92); transition: opacity .3s ease, transform .3s ease;
}}
body:has([data-testid="stStatusWidgetRunningIcon"]) .paddy-loader {{ opacity: 1; transform: translate(-50%, -50%) scale(1); }}
.paddy-loader svg {{ display: block; }}
.paddy-loader .stalk {{ transform-box: fill-box; transform-origin: 50% 100%; animation: rise 1.4s ease-in-out infinite; }}
.paddy-loader .stalk path {{ stroke: var(--leaf); stroke-width: 1.8; fill: none; stroke-linecap: round; }}
.paddy-loader .grain {{ fill: #d4a017; }}
.paddy-loader .stalk:nth-child(2) {{ animation-delay: .14s; }}
.paddy-loader .stalk:nth-child(3) {{ animation-delay: .28s; }}
.paddy-loader .stalk:nth-child(4) {{ animation-delay: .42s; }}
.paddy-loader .stalk:nth-child(5) {{ animation-delay: .56s; }}
.paddy-loader .soil {{ stroke: #d8bfa3; stroke-width: 2; stroke-linecap: round; }}
@keyframes rise {{
  0%, 60%, 100% {{ opacity: .25; transform: scaleY(.6) rotate(-4deg); }}
  30% {{ opacity: 1; transform: scaleY(1.05) rotate(5deg); }}
}}
</style>
{loader_html()}
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ top bar
with st.container(key="topbar"):
    c_brand, c_lang, c_font, c_hist, c_new = st.columns([7.6, 1.75, 0.75, 1.9, 0.55], vertical_alignment="center",
                                                        gap="small")
    c_brand.markdown(f"""
      <div class="brand"><div class="logo">🌾</div><div>
        <div class="name">{t['app_name']}<span class="alt">{t['app_alt']}</span></div>
        <div class="tag">{t['tagline']}</div></div></div>""", unsafe_allow_html=True)
    c_lang.segmented_control("Language", ["en", "ta"], default=lang, key="lang_ctl",
                             format_func=lambda x: "English" if x == "en" else "தமிழ்",
                             label_visibility="collapsed")
    with c_font.popover("Aa", help=t["font"], width="stretch"):
        st.segmented_control(t["font"], list(FONT_SIZES), key="font_ctl", on_change=set_font,
                             default=next((k for k, v in FONT_SIZES.items() if v == ss.font_scale), "m"),
                             format_func=lambda k: t["sizes"][k])
    with c_hist.popover(t["history"], icon=":material/history:", width="stretch"):
        past = [(cid, c) for cid, c in reversed(ss.chats.items()) if c["messages"]]
        if not past:
            st.caption(t["no_history"])
        for cid, c in past:
            st.button(f"{c['title'] or t['untitled']}  ·  {c['created']}", key=f"hist_{cid}",
                      on_click=switch_chat, args=(cid,), width="stretch",
                      type="primary" if cid == ss.chat_id else "secondary")
    c_new.button("", icon=":material/edit_square:", help=t["new_chat"], on_click=new_chat,
                 width="stretch", key="new_chat_btn")

st.write("")
left, right = st.columns([11, 9], gap="medium")

# ------------------------------------------------------------------ left: directory
with left, st.container(key="catalog"):
    cats = catalog["categories"]
    schemes = catalog["schemes"]

    h1, h2 = st.columns([1.4, 1], vertical_alignment="bottom")
    query = h2.text_input("search", placeholder="🔍 " + t["search"], label_visibility="collapsed")
    level = st.segmented_control("Level", ["all", "State", "Central"], default="all", key="level_ctl",
                                 format_func=lambda x: t[x.lower()], label_visibility="collapsed") or "all"
    cat_keys = ["all", "recent", "popular", *cats]
    cat_label = {"all": f"🌾 {t['all']}", "recent": f"🆕 {t['recent']}", "popular": f"⭐ {t['popular']}",
                 **{k: f"{v['icon']} {v[lang]}" for k, v in cats.items()}}
    cat = st.pills("Category", cat_keys, default="all", key="cat_ctl", format_func=cat_label.get,
                   label_visibility="collapsed") or "all"

    def visible(s):
        if level != "all" and level not in s["level"]:
            return False
        if cat in ("recent", "popular") and cat not in s["tags"]:
            return False
        if cat in cats and s["category"] != cat:
            return False
        blob = " ".join([s["id"], s["name_en"], s["name_ta"], s["desc_en"], s["desc_ta"]]).lower()
        return query.strip().lower() in blob

    shown = [s for s in schemes if visible(s)]
    h1.markdown(f"<p class='panel-title'>📚 {t['directory']}</p>"
                f"<div class='panel-sub'>{t['directory_sub'].format(n=len(shown))}</div>",
                unsafe_allow_html=True)

    with st.container(height=430, border=False, key="cardgrid"):
        if not shown:
            st.info(t["no_match"])
        grid = st.columns(2, gap="small")
        for i, s in enumerate(shown):
            name, alt = (s["name_ta"], s["name_en"]) if lang == "ta" else (s["name_en"], s["name_ta"])
            lvl_cls = {"Central": "central", "State": "state"}.get(s["level"], "both")
            badges = f"<span class='badge {lvl_cls}'>{t['lvl'][s['level']]}</span>"
            if "recent" in s["tags"]:
                badges += f"<span class='badge recent' title='{html.escape(s.get('basis', ''))}'>🆕 {t['recent']}</span>"
            if "popular" in s["tags"]:
                badges += f"<span class='badge popular'>⭐ {t['popular']}</span>"
            with grid[i % 2].container(border=True, key=f"card_{s['id']}"):
                st.markdown(f"""<div class="card">
                    <div class="top">{cats[s['category']]['icon']} {badges}</div>
                    <div class="nm">{html.escape(name)}</div><div class="alt">{html.escape(alt)}</div>
                    <div class="ds">{html.escape(s['desc_' + lang])}</div></div>""", unsafe_allow_html=True)
                label = f"{s['name_ta']} ({s['name_en']})" if lang == "ta" else s["name_en"]
                st.button(t["ask"], key=f"ask_{s['id']}", on_click=ask, args=(t["scheme_q"].format(name=label),))

    with st.container(key="quickq"):
        st.markdown(f"<div class='section-h'>💬 {t['quick']}</div>", unsafe_allow_html=True)
        qcols = st.columns(2, gap="small")
        for i, q in enumerate(catalog["sample_questions"][lang]):
            qcols[i % 2].button(q, key=f"q_{lang}_{i}", on_click=ask, args=(q,), width="stretch")

    st.markdown(f"""<div class="heritage"><div class="k">{KURAL}</div><div class="m">திருக்குறள் 1031 · {t['kural_meaning']}</div></div>""",
                unsafe_allow_html=True)


# ------------------------------------------------------------------ right: chat
def render_message(m):
    with st.chat_message(m["role"], avatar="🧑‍🌾" if m["role"] == "user" else "🌾"):
        if m.get("voice"):
            st.caption(f"🎤 {t['voice_note']}")
        st.markdown(m["content"])
        if m.get("sources"):
            st.markdown(f"<div class='src'>📚 {t['sources']}: {html.escape(' · '.join(m['sources']))}</div>",
                        unsafe_allow_html=True)
        if m.get("audio"):
            st.audio(m["audio"], format="audio/mp3")


def answer(question: str, by_voice: bool, box):
    history = list(chat["messages"])
    user_msg = {"role": "user", "content": question, "voice": by_voice}
    chat["messages"].append(user_msg)
    chat["title"] = chat["title"] or question[:48]
    with box:
        render_message(user_msg)
        with st.chat_message("assistant", avatar="🌾"):
            msg, meta = {"role": "assistant", "content": ""}, {"language": lang}
            try:
                with st.spinner(t["searching"]):
                    tokens = stream_chat(question, history, meta)
                    first = next(tokens, "")  # wait for retrieval before streaming starts
                msg["content"] = st.write_stream(token for part in ([first], tokens) for token in part)
                lang_q = meta["language"]
                msg["sources"] = meta.get("sources", [])
                st.markdown(f"<div class='src'>📚 {T[lang_q]['sources']}: "
                            f"{html.escape(' · '.join(msg['sources']))}</div>", unsafe_allow_html=True)
                if by_voice:
                    with st.spinner(T[lang_q]["speaking"]):
                        msg["audio"] = api_speak(msg["content"], lang_q)
                    st.audio(msg["audio"], format="audio/mp3", autoplay=True)
            except Exception as e:  # show the error in-chat instead of crashing the app
                msg["content"] = t["error"].format(e=e)
                st.error(msg["content"])
    chat["messages"].append(msg)


with right, st.container(key="chatpane"):
    h_title, h_clear = st.columns([5, 1.2], vertical_alignment="top")
    h_title.markdown(f"<p class='panel-title'>🌱 {t['chat_title']}</p><div class='panel-sub'>{t['chat_sub']}</div>",
                     unsafe_allow_html=True)
    h_clear.button(t["clear"], icon=":material/delete_sweep:", on_click=clear_chat, type="tertiary",
                   width="stretch", key="clear_btn", disabled=not chat["messages"])

    box = st.container(height=440, border=False, key="chatbox")
    with box:
        if not chat["messages"]:
            with st.chat_message("assistant", avatar="🌾"):
                st.markdown(t["welcome"])
        for m in chat["messages"]:
            render_message(m)
        if not ready:
            st.warning(t["no_index"])

    typed = st.chat_input(t["placeholder"], disabled=not ready)
    st.markdown(f"<div class='voice-hint'>{t['voice_label']}</div>", unsafe_allow_html=True)
    audio = st.audio_input(t["voice_label"], key=f"audio_{ss.audio_key}", disabled=not ready,
                           label_visibility="collapsed")

    question, by_voice = None, False
    if ss.pending and ready:
        question, ss.pending = ss.pending, None
    elif typed:
        question = typed
    elif audio is not None:
        with box, st.spinner(t["transcribing"]):
            question = api_transcribe(audio.getvalue())
        by_voice = True
        ss.audio_key += 1  # fresh recorder widget on the next run
    if question:
        answer(question, by_voice, box)

st.markdown(f"<div class='disclaimer'>⚠️ {t['disclaimer']}</div>", unsafe_allow_html=True)
