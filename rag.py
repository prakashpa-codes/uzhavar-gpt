"""LangChain RAG pipeline: FAISS (read-only) + OpenAI mini model.

Flow per question:
  1. Rewrite the question (+ recent chat) into a standalone English search query.
     Most of the corpus is English, so this makes Tamil questions retrieve well.
     A first English question is already standalone, so we skip that LLM call.
  2. Search with both the rewritten query and the original question, using FAISS
     (meaning) and BM25 (exact words like PMKSY or a GO number). If the query names a
     scheme acronym or a department, also run a FAISS search filtered to it.
     All these ranked lists are merged with reciprocal rank fusion (RRF).
  3. Stream an answer grounded only in the retrieved context, in the asked language.

LangSmith tracing is automatic when LANGSMITH_TRACING=true is set in .env.
"""

import re

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import CHAT_MODEL, EMBED_MODEL, INDEX_DIR, SOURCE_TITLES, TOP_K

TAMIL = re.compile(r"[஀-௿]")


def detect_language(text: str) -> str:
    return "ta" if TAMIL.search(text or "") else "en"


def tokenize(text: str) -> list[str]:
    # Tamil vowel signs aren't "\w" in Python, so add the Tamil block explicitly,
    # otherwise every Tamil word gets chopped into pieces.
    return re.findall(r"[\w஀-௿]+", text.lower())


# Words a farmer (or the rewriter) might use for each department -> the "department"
# value ingest.py stores. Plain "agriculture" is left out since nearly everything matches it.
DEPARTMENTS = {
    "Horticulture": ["horticulture", "தோட்டக்கலை"],
    "Agricultural Engineering": ["agricultural engineering", "agri engineering", "வேளாண் பொறியியல்"],
    "Marketing & Agri Business": ["marketing", "agri business", "agribusiness", "வேளாண் விற்பனை", "வேளாண் வணிக"],
    "Seed Certification & Organic Certification": ["seed certification", "organic certification", "சான்றளிப்பு"],
    "Sugarcane": ["sugarcane department", "sugar department", "சர்க்கரைத் துறை"],
    "Tamil Nadu Agricultural University": ["tnau", "agricultural university", "வேளாண்மை பல்கலைக்கழகம்"],
}


def rrf(rankings: list[list], k: int = 60) -> list:
    """Reciprocal rank fusion: a chunk ranked high in any list (or present in many) wins."""
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, d in enumerate(ranking):
            key = (d.metadata.get("source"), d.page_content[:120])
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            docs.setdefault(key, d)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question as one standalone English search query about "
     "Tamil Nadu / Central government agriculture schemes. Resolve references using the "
     "chat history, translate Tamil to English, keep scheme names and acronyms. "
     "Return only the query."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are UzhavarGPT, a friendly assistant helping Tamil Nadu farmers understand "
     "government agriculture schemes (Tamil Nadu State and Central schemes).\n\n"
     "Rules:\n"
     "- Answer ONLY from the context below. If the context does not contain the answer, say so "
     "honestly and suggest contacting the nearest Agricultural Extension Centre "
     "(வேளாண் விரிவாக்க மையம்) or the Kisan Call Centre (1800-180-1551).\n"
     "- Reply entirely in {language}. Keep scheme names recognisable (add the English name/acronym in brackets when replying in Tamil).\n"
     "- Use simple words a farmer understands. Structure with short markdown headings and bullets where useful: "
     "Benefits / Subsidy, Eligibility, Documents required, How to apply.\n"
     "- Quote amounts, percentages and limits exactly as in the context, and mention the year if the context gives one. "
     "Never mix figures from different schemes or passages into one statement.\n"
     "- Be concise (under ~250 words) unless the user asks for detail. Do not invent URLs or phone numbers.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


class SchemeRAG:
    def __init__(self):
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        # Our own locally built index, so pickle deserialization is safe here.
        self.store = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

        # BM25 over the very same chunks FAISS holds, so no separate index to keep in sync.
        # (docstore._dict is private, but it's the only way LangChain exposes the chunks.)
        chunks = list(self.store.docstore._dict.values())
        self.bm25 = BM25Retriever.from_documents(chunks, k=TOP_K, preprocess_func=tokenize)

        # Acronym -> scheme titles, e.g. "PMKSY" -> both PMKSY micro-irrigation sections.
        # Only all-caps words of 3+ letters, so normal words in titles don't count.
        # Hyphens split them, so "PMKSY-MICROIRRIGATION" is also found by "PMKSY".
        self.acronyms = {}
        for scheme in {c.metadata["scheme"] for c in chunks if "scheme" in c.metadata}:
            for acr in re.findall(r"\b[A-Z]{3,}\b", scheme):
                self.acronyms.setdefault(acr, set()).add(scheme)
        self.has_departments = any("department" in c.metadata for c in chunks)

        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
        self.rewriter = (REWRITE_PROMPT | llm | StrOutputParser()).with_config(run_name="rewrite_query")
        self.answerer = (ANSWER_PROMPT | llm | StrOutputParser()).with_config(run_name="answer")

    @staticmethod
    def _history(messages: list[dict], limit: int = 6):
        out = []
        for m in messages[-limit:]:
            cls = HumanMessage if m["role"] == "user" else AIMessage
            out.append(cls(content=m["content"]))
        return out

    def metadata_filter(self, text: str):
        """Return a FAISS filter if the query clearly names a scheme acronym or a department."""
        schemes = set()
        for acr, titles in self.acronyms.items():
            # Case-sensitive on purpose: "FIG" is a scheme, "fig" is a fruit.
            if re.search(rf"\b{re.escape(acr)}\b", text):
                schemes |= titles
        lowered = text.lower()
        depts = {d for d, words in DEPARTMENTS.items() if any(w in lowered for w in words)} if self.has_departments else set()
        if not schemes and not depts:
            return None
        return lambda m: m.get("scheme") in schemes or m.get("department") in depts

    def retrieve(self, question: str, history: list[dict]):
        # A first English question is already a standalone English query, so the rewrite
        # would only add latency. Tamil needs translating, and follow-ups need the history.
        if history or detect_language(question) == "ta":
            query = self.rewriter.invoke({"question": question, "history": self._history(history)})
        else:
            query = question

        rankings = []
        for q in dict.fromkeys([query, question]):  # dedupe if identical
            rankings.append(self.store.similarity_search(q, k=TOP_K))
            rankings.append(self.bm25.invoke(q))

        # The filtered search is just one more list in the fusion, not a hard filter,
        # so a wrong guess can only add a few chunks, never hide the right ones.
        # fetch_k is large because FAISS filters *after* fetching the nearest candidates.
        doc_filter = self.metadata_filter(f"{query}\n{question}")
        if doc_filter:
            rankings.append(self.store.similarity_search(query, k=TOP_K, filter=doc_filter, fetch_k=400))

        return query, rrf(rankings)[: TOP_K + 2]

    def stream_answer(self, question: str, history: list[dict], docs, language: str):
        context = "\n\n---\n\n".join(
            f"[{SOURCE_TITLES.get(d.metadata.get('source'), d.metadata.get('title'))}{', p.' + str(d.metadata['page']) if 'page' in d.metadata else ''}]\n{d.page_content}"
            for d in docs
        )
        return self.answerer.stream({
            "question": question,
            "history": self._history(history),
            "context": context,
            "language": "Tamil (தமிழ்)" if language == "ta" else "English",
        })


def format_sources(docs, limit: int = 4) -> list[str]:
    """Top unique, human-readable citations like 'TNAGRISNET scheme portal - PM-KISAN' (docs are rank-ordered)."""
    out = []
    for d in docs:
        if len(out) == limit:
            break
        m = d.metadata
        label = SOURCE_TITLES.get(m.get("source"), m.get("title", m.get("source", "")))
        if "page" in m:
            label += f" (p. {m['page']})"
        elif "scheme" in m:
            label += f" - {m['scheme'][:60]}"
        if label not in out:
            out.append(label)
    return out
