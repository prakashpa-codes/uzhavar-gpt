"""LangChain RAG pipeline: FAISS (read-only) + OpenAI mini model.

Flow per question:
  1. Rewrite the question (+ recent chat) into a standalone English search query.
     Most of the corpus is English, so this makes Tamil questions retrieve well.
  2. Search FAISS with both the rewritten query and the original question, merge.
  3. Stream an answer grounded only in the retrieved context, in the asked language.

LangSmith tracing is automatic when LANGSMITH_TRACING=true is set in .env.
"""

import re

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import CHAT_MODEL, EMBED_MODEL, INDEX_DIR, SOURCE_TITLES, TOP_K

TAMIL = re.compile(r"[஀-௿]")


def detect_language(text: str) -> str:
    return "ta" if TAMIL.search(text or "") else "en"


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

    def retrieve(self, question: str, history: list[dict]):
        query = self.rewriter.invoke({"question": question, "history": self._history(history)})
        docs, seen = [], set()
        for q in dict.fromkeys([query, question]):  # dedupe if identical
            for d in self.store.similarity_search(q, k=TOP_K):
                key = (d.metadata.get("source"), d.page_content[:120])
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
        return query, docs[: TOP_K + 2]

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
