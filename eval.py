"""Small evaluation set for UzhavarGPT retrieval + answers.

    python eval.py                 # all questions, retrieval + answer + LLM judge
    python eval.py --no-judge      # retrieval only (fast, cheap)

For each question we know which source file(s) the answer lives in and a keyword the
right chunk must contain. A retrieved chunk counts as relevant only when both match,
so "hit" means we found the right passage, not just the right PDF.

Metrics (written to eval_results.csv, summary printed at the end):
  hit      - a relevant chunk is anywhere in the retrieved context (hit-rate)
  rr       - 1 / rank of the first relevant chunk (averaged = MRR)
  faithful - LLM judge, 1-5: is every claim in the answer backed by the context?
"""

import argparse
import csv
import re
import sys
import time

from langchain_openai import ChatOpenAI

from config import CHAT_MODEL, ROOT
from rag import SchemeRAG, detect_language

PN, VOL1, VOL2 = "agri_e_pn_2026_27.pdf", "BookIIISchemesVolIF.pdf", "BookIIISchemesVolIIF.pdf"
SCHEMES, GOSCHEME, TNGOV = "tnagrisnet_schemes.md", "tnagrisnet_goscheme_subsidies.md", "tn_gov_agri_schemes.md"

# Each keyword was checked against source-data/ so the expected answer really is there.
CASES = [
    # English
    {"q": "What is the subsidy for a rotavator under farm mechanisation?", "keywords": ["rotavator"], "sources": [GOSCHEME, SCHEMES]},
    {"q": "How much subsidy does PMKSY give for micro irrigation?", "keywords": ["PMKSY"], "sources": [SCHEMES, GOSCHEME, PN, VOL1]},
    {"q": "How much money does a farmer get under PM-KISAN?", "keywords": ["PM-KISAN"], "sources": [PN, VOL1]},
    {"q": "Who can apply for a Kisan Credit Card?", "keywords": ["Kisan Credit"], "sources": [VOL1, PN]},
    {"q": "How do I get an NPOP organic certificate?", "keywords": ["NPOP"], "sources": [SCHEMES, GOSCHEME]},
    {"q": "What support is given under the Palmyrah Development Mission?", "keywords": ["Palmyrah"], "sources": [SCHEMES, GOSCHEME, PN, VOL2]},
    {"q": "How can farmers sell their produce through e-NAM?", "keywords": ["e-NAM"], "sources": [SCHEMES, GOSCHEME, PN, VOL1]},
    {"q": "What is the Farmers Interest Group scheme?", "keywords": ["Farmers Interest Group"], "sources": [TNGOV]},
    {"q": "Is there a scheme for zinc sulphate and gypsum for paddy?", "keywords": ["zinc sulphate"], "sources": [SCHEMES]},
    {"q": "What is Uzhavar Sandhai?", "keywords": ["Uzhavar Sandhai"], "sources": [PN, VOL2]},
    {"q": "What are the benefits under Paramparagat Krishi Vikas Yojana?", "keywords": ["Paramparagat", "PKVY"], "sources": [SCHEMES, GOSCHEME, VOL1, PN]},
    {"q": "Which schemes give subsidy for solar pumps?", "keywords": ["solar"], "sources": [GOSCHEME, SCHEMES, PN, VOL2]},
    # Follow-up: only makes sense with the history, tests the rewrite step
    {"q": "What documents do I need to apply for it?", "keywords": ["PMKSY", "micro irrigation", "microirrigation"], "sources": [SCHEMES, GOSCHEME],
     "history": [{"role": "user", "content": "Tell me about PMKSY micro irrigation subsidy"},
                 {"role": "assistant", "content": "PMKSY gives subsidy for drip and sprinkler irrigation."}]},
    # Tamil
    {"q": "பிரதம மந்திரி கிசான் திட்டத்தில் எவ்வளவு பணம் கிடைக்கும்?", "keywords": ["PM-KISAN"], "sources": [PN, VOL1]},
    {"q": "சொட்டு நீர் பாசனத்திற்கு எவ்வளவு மானியம் கிடைக்கும்?", "keywords": ["micro irrigation", "microirrigation", "drip"], "sources": [SCHEMES, GOSCHEME, PN, VOL1, VOL2]},
    {"q": "பயிர் காப்பீடு திட்டம் பற்றி சொல்லுங்கள்", "keywords": ["Fasal Bima", "crop insurance"], "sources": [PN, VOL1]},
    {"q": "இலவச மரக்கன்றுகள் எப்படி பெறுவது?", "keywords": ["Tree Saplings", "மரக் கன்று"], "sources": [SCHEMES]},
    {"q": "உழவர் சந்தை என்றால் என்ன?", "keywords": ["Uzhavar Sandhai"], "sources": [PN, VOL2]},
    {"q": "மண் வள அட்டை திட்டம் என்றால் என்ன?", "keywords": ["Soil Health Card"], "sources": [PN, VOL1]},
    {"q": "தோட்டக்கலை துறையின் தேசிய தோட்டக்கலை இயக்கம் மூலம் என்ன மானியம் கிடைக்கும்?", "keywords": ["National Horticulture Mission", "NHM"], "sources": [SCHEMES, GOSCHEME, PN, VOL1]},
]

JUDGE_PROMPT = """You are checking an AI answer for faithfulness to its source context.

Context:
{context}

Answer:
{answer}

Score how well every factual claim in the answer (amounts, eligibility, steps, names)
is supported by the context. Saying "not found, contact the Kisan Call Centre" counts
as faithful if the context really lacks the answer.
5 = fully supported, 3 = partly supported, 1 = mostly unsupported or invented.
Reply with only the number."""


def is_relevant(doc, case) -> bool:
    text = doc.page_content.lower()
    return doc.metadata.get("source") in case["sources"] and any(k.lower() in text for k in case["keywords"])


def judge(llm, docs, answer) -> int | None:
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    reply = llm.invoke(JUDGE_PROMPT.format(context=context, answer=answer)).content
    match = re.search(r"[1-5]", reply)
    return int(match.group()) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true", help="skip answer generation and the LLM judge")
    parser.add_argument("--out", default=str(ROOT / "eval_results.csv"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles choke on Tamil otherwise

    rag = SchemeRAG()
    judge_llm = None if args.no_judge else ChatOpenAI(model=CHAT_MODEL, temperature=0)
    rows = []

    for i, case in enumerate(CASES, start=1):
        history = case.get("history", [])
        language = detect_language(case["q"])

        start = time.time()
        query, docs = rag.retrieve(case["q"], history)
        retrieve_s = time.time() - start

        rank = next((r for r, d in enumerate(docs, start=1) if is_relevant(d, case)), None)
        row = {
            "id": i,
            "lang": language,
            "question": case["q"],
            "search_query": query,
            "hit": int(rank is not None),
            "rank": rank or "",
            "rr": round(1 / rank, 3) if rank else 0,
            "retrieve_s": round(retrieve_s, 2),
            "top_sources": " | ".join(f"{d.metadata.get('source')}:{d.metadata.get('page', d.metadata.get('scheme', ''))}"[:70] for d in docs[:3]),
        }

        if judge_llm:
            answer = "".join(rag.stream_answer(case["q"], history, docs, language))
            row["faithful"] = judge(judge_llm, docs, answer)
            row["answer"] = answer.replace("\n", " ")[:500]

        rows.append(row)
        print(f"[{i:2}/{len(CASES)}] {language} hit={row['hit']} rank={row['rank'] or '-'} "
              f"faithful={row.get('faithful', '-')} {case['q'][:60]}")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows Tamil
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"\nHit-rate: {sum(r['hit'] for r in rows) / n:.2f}   MRR: {sum(r['rr'] for r in rows) / n:.3f}   "
          f"avg retrieve: {sum(r['retrieve_s'] for r in rows) / n:.2f}s")
    for lang in ("en", "ta"):
        part = [r for r in rows if r["lang"] == lang]
        print(f"  {lang}: hit-rate {sum(r['hit'] for r in part) / len(part):.2f}, MRR {sum(r['rr'] for r in part) / len(part):.3f}")
    scores = [r["faithful"] for r in rows if r.get("faithful")]
    if scores:
        print(f"Faithfulness (1-5): {sum(scores) / len(scores):.2f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
