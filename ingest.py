"""Build the local FAISS index from everything in source-data/.

    python ingest.py            # parse, chunk, embed, save to faiss_index/
    python ingest.py --dry-run  # parse + chunk only, print stats (no API calls)

PDFs: the ICAR "Schemes Made Simple" books print Tamil in a legacy font that pypdf
extracts as mojibake, but every scheme is also written in English. So for PDFs we keep
clean English lines and drop garbled / Tamil-script lines. Clean bilingual Tamil comes
from the scraped portal markdown files (see scrape_schemes.py), which are kept as-is.
"""

import argparse
import json
import re
import time

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBED_MODEL, INDEX_DIR, SOURCE_TITLES

TAMIL = re.compile(r"[஀-௿]")
LATIN1_JUNK = re.compile(r"[À-ÿŒ-˿]")  # legacy-font mojibake (Ö¼ïòÂ...)


def is_clean_line(line: str) -> bool:
    if "�" in line or TAMIL.search(line):
        return False
    junk = len(LATIN1_JUNK.findall(line))
    return junk == 0 or junk / max(len(line), 1) < 0.05


def load_pdf(path) -> list[Document]:
    docs = []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        lines = [l.strip() for l in (page.extract_text() or "").splitlines()]
        text = "\n".join(l for l in lines if l and is_clean_line(l))
        text = re.sub(r"[ \t]{2,}", " ", text)
        if len(text) > 80:  # skip covers, blank and pure-Tamil pages
            docs.append(Document(text, metadata={"source": path.name, "page": page_no}))
    return docs


def load_markdown(path) -> list[Document]:
    """One Document per '## scheme' section, so chunks never mix two schemes."""
    text = path.read_text(encoding="utf-8")
    docs = []
    for section in re.split(r"\n(?=## )", text):
        if not section.startswith("## "):
            continue
        title = section.splitlines()[0][3:].strip()
        meta = {"source": path.name, "scheme": title}
        # The scraper writes the department as "Department: Horticulture / தோட்டக்கலை துறை"
        # (TNAGRISNET) or "**Concerned Department:** ..." (tn.gov.in). Keep the English part
        # so rag.py can narrow the search to one department.
        dept = re.search(r"^(?:Department:|- \*\*Concerned Department:\*\*)\s*([^/\n]+)", section, re.M)
        if dept:
            meta["department"] = dept.group(1).strip()
        docs.append(Document(section.strip(), metadata=meta))
    return docs


def load_all() -> list[Document]:
    docs = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.name == "SOURCES.md":  # list of links, not content
            continue
        suffix = path.suffix.lower()
        loaded = load_pdf(path) if suffix == ".pdf" else load_markdown(path) if suffix in {".md", ".txt"} else []
        for d in loaded:
            d.metadata["title"] = SOURCE_TITLES.get(path.name, path.stem)
        print(f"  {path.name}: {len(loaded)} sections/pages")
        docs.extend(loaded)
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk only")
    args = parser.parse_args()

    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_all()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    # Prefix scheme name to markdown chunks so split sections keep their context.
    for c in chunks:
        if "scheme" in c.metadata and not c.page_content.startswith("## "):
            c.page_content = f"## {c.metadata['scheme']}\n{c.page_content}"
    total_chars = sum(len(c.page_content) for c in chunks)
    print(f"{len(docs)} docs -> {len(chunks)} chunks ({total_chars:,} chars)")

    if args.dry_run:
        for c in chunks[::max(len(chunks) // 4, 1)][:4]:
            print("-" * 60, c.metadata, c.page_content[:300], sep="\n")
        return

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    batch, store = 200, None
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        if store is None:
            store = FAISS.from_documents(part, embeddings)
        else:
            store.add_documents(part)
        print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}")

    store.save_local(str(INDEX_DIR))
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "embed_model": EMBED_MODEL,
        "chunks": len(chunks),
        "sources": sorted({c.metadata["source"] for c in chunks}),
    }
    (INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved FAISS index -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
