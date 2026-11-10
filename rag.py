"""A STAND-IN for the real RAG system, so the UI can be built and tested first.

It searches the real 5,520-paper arXiv cs.CL sample in data/ with simple word
matching (no embeddings, no LLM, no torch). The important part is NOT how it
searches but WHAT it returns: the "contract" that the UI depends on.

Later, replace the body of ask() with a call to the real pipeline
(e.g. a rag_core.ask() like the one in INTEGRATION.md) and convert its output to the same shape.
As long as the shape stays the same, app.py doesn't need to change.

The contract:
{
  "query":   str,
  "summary": str,                 # written by the LLM in the real system
  "papers": [                     # ALWAYS from retrieval, never written by the LLM
    {"arxiv_id", "title", "authors": [str], "year": int, "categories": [str],
     "abstract", "abstract_snippet", "why_relevant", "score": float, "abs_url", "pdf_url"}
  ]
}

Try it on its own:  .venv/Scripts/python.exe -X utf8 rag.py
"""

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "arxiv_cs_cl_sample.jsonl"
TOP_K = 5

STOPWORDS = set("""a an and are as at be by can do does for from how i in is it of on or
paper papers the to what which with about using use""".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if w not in STOPWORDS}


def load_papers() -> list[dict]:
    papers = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            p["_title_words"] = words(p["title"])
            p["_abstract_words"] = words(p["abstract"])
            papers.append(p)
    return papers


PAPERS = load_papers()  # loaded once, when the app starts


def split_authors(authors: str) -> list[str]:
    # "A, B and C" -> ["A", "B", "C"]
    parts = re.split(r",\s*|\s+and\s+", authors.replace("\n", " "))
    return [a.strip() for a in parts if a.strip()]


def year_from_id(arxiv_id: str) -> int:
    # New-style arXiv IDs start with YYMM: "2302.00129" -> 2023
    return 2000 + int(arxiv_id[:2])


def ask(question: str, history: list[dict] | None = None) -> dict:
    # history: past turns as [{"role": "user"|"assistant", "content": str}].
    # This placeholder ignores it. The real RAG can use it for follow-up questions.
    q = words(question)
    scored = []
    for p in PAPERS:
        # A word in the title counts double: titles are short and on-topic.
        hits_title = q & p["_title_words"]
        hits_abstract = q & p["_abstract_words"]
        score = (2 * len(hits_title) + len(hits_abstract)) / (3 * len(q)) if q else 0
        if score > 0:
            scored.append((score, p, sorted(hits_title | hits_abstract)))
    scored.sort(key=lambda x: -x[0])

    papers = []
    for score, p, matched in scored[:TOP_K]:
        papers.append({
            "arxiv_id": p["id"],
            "title": " ".join(p["title"].split()),
            "authors": split_authors(p["authors"]),
            "year": year_from_id(p["id"]),
            "categories": p["categories"].split(),
            "abstract": " ".join(p["abstract"].split()),            # full, goes into the BibTeX export
            "abstract_snippet": " ".join(p["abstract"].split())[:300] + "...",  # short, for the chat card
            # In the real system, the LLM writes one sentence here.
            "why_relevant": "Matches: " + ", ".join(matched),
            "score": round(score, 3),
            "abs_url": f"https://arxiv.org/abs/{p['id']}",
            "pdf_url": f"https://arxiv.org/pdf/{p['id']}",
        })

    summary = (f"Found {len(papers)} papers for this question. "
               "(Prototype: in the real system, the LLM writes a short summary here.)"
               if papers else "No matching papers found. Try other words.")
    return {"query": question, "summary": summary, "papers": papers}


if __name__ == "__main__":
    result = ask("retrieval augmented generation evaluation")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
