# Connecting your RAG to the Chainlit frontend

The frontend (chat, paper cards with **Add**, the Reading List panel, export to zip, chat history) is done and runs on its own with a placeholder search. To connect your RAG system, you change **one function**, `ask()` in `rag.py`. Nothing else has to change.

```
app.py (UI) ──calls──► rag.ask(question, history) ──returns──► the JSON below
                              ▲
                     only this changes: call your pipeline here
```

## 1. The contract: what `ask()` must return

```python
{
  "query":   "graph neural networks for traffic forecasting",
  "summary": "Most work combines spatial GNN layers with temporal models [1709.04875] ...",
  "papers": [                       # 0–5 items, best first
    {
      "arxiv_id":         "1709.04875",                    # required, plain ID without version
      "title":            "Spatio-Temporal Graph ...",     # required
      "authors":          ["Bing Yu", "Haoteng Yin"],      # required, a LIST of names
      "year":             2017,                            # required, int
      "categories":       ["cs.LG", "stat.ML"],            # required, first one is used in BibTeX
      "abstract":         "We propose STGCN, a novel ...", # full abstract, goes into the BibTeX export
      "abstract_snippet": "We propose STGCN ...",          # ~300 chars, shown on the chat card
      "why_relevant":     "Introduces the GNN + temporal convolution baseline.",  # may be ""
      "score":            0.83,
      "abs_url":          "https://arxiv.org/abs/1709.04875",
      "pdf_url":          "https://arxiv.org/pdf/1709.04875"
    }
  ]
}
```

Rules:
- **`papers` must come from retrieval (the Qdrant payload), never from LLM output.** The export downloads `pdf_url` and builds BibTeX from these fields, so an invented ID breaks it. The LLM only writes `summary`, and optionally `why_relevant`.
- **One entry per paper.** If your index has several chunks per paper, group by `arxiv_id` first, e.g. with Qdrant's `query_points_groups(..., group_by="arxiv_id")`.
- **Nothing relevant** (your gate refuses)? Return `"papers": []` and put the refusal text in `summary`. The UI handles it.
- **LLM down or out of quota?** Still return the papers, with a short `summary` such as "Here are the most relevant papers." The reading list still works.

## 2. Example: adapter for a `rag_core.py` pipeline

If your pipeline is the `rag_core.ask()` from our earlier Streamlit MVP (`paper-assistant-mvp`), this is roughly all it takes. It's **untested**: the field names come from `ingest.py` and `rag_core.py`.

```python
# rag.py
import re
import rag_core   # copy rag_core.py (and qdrant_db/) next to app.py, or add its folder to sys.path

CLIENT = rag_core.load_client()      # load once at startup, not per question
EMBEDDER = rag_core.load_embedder()

def why_from_answer(answer: str, arxiv_id: str) -> str:
    """Use the answer sentence that cites this paper as its 'why relevant' line."""
    for sentence in re.split(r"(?<=[.!?])\s+", answer or ""):
        if arxiv_id in sentence:
            return re.sub(r"\s*\[[^\]]*\]", "", sentence).strip()
    return ""

def ask(question: str, history=None) -> dict:
    trace = rag_core.ask(CLIENT, EMBEDDER, question)

    if trace["status"] == "refused":
        return {"query": question, "papers": [],
                "summary": "I couldn't find papers on this in the collection. Try rephrasing."}

    papers = []
    for p in trace["shown"][:5]:
        abstract = p["text"].split("\n\n", 1)[-1]   # the payload text is "title\n\nabstract"
        papers.append({
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "authors": [a.strip() for a in re.split(r",\s*|\s+and\s+", p["authors"]) if a.strip()],
            "year": p["year"],
            "categories": p["categories"],
            "abstract": abstract,
            "abstract_snippet": abstract[:300] + "...",
            "why_relevant": why_from_answer(trace.get("answer"), p["arxiv_id"]),
            "score": round(p["final"], 3),
            "abs_url": p["url"],
            "pdf_url": p["url"].replace("/abs/", "/pdf/"),
        })

    summary = trace["answer"] if trace["status"] == "answered" else \
        "Answer generation is busy right now. Here are the most relevant papers."
    return {"query": question, "summary": summary, "papers": papers}
```

## 3. Conversation memory (for follow-up questions)

The app already keeps the chat as a list and saves it with each chat in the history:

```python
history = [{"role": "user", "content": "GAN papers"},
           {"role": "assistant", "content": "Found 5 papers...\n1. [2306.16313] An Adversarial ...\n2. ..."}]
```

It holds the last 10 turns, and the assistant turns include the numbered titles and IDs. `app.py` already passes it in as `rag.ask(question, history=...)`, so you only need to use it inside `ask()`. In LangChain, the list works as it is:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's last question into a standalone search query."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
standalone = (prompt | llm | StrOutputParser()).invoke({"history": history, "question": question})
# then retrieve with `standalone` instead of `question`
```

Main use: "more papers like the second one" → "graph neural networks for traffic forecasting", **before** retrieval.

## 4. Running it

```powershell
cd arxiv-paper-assistant
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # + your RAG's requirements
.venv\Scripts\chainlit.exe run app.py -w
```

- Create `.env` first (it isn't in git; see the README). Login is any username + the password in `.env`.
- Test your `ask()` alone first: `python -X utf8 rag.py` should print valid JSON with 5 papers. Then start the app.
- Load models and the Qdrant client **once** at module level, as in the example. Chainlit imports `rag.py` once, so every question reuses them.

## 5. Things to agree on together

| Topic | Current state | Decide |
|---|---|---|
| Reading list scope | One list **per chat** | Or one list **per user** across all chats (more like Zotero)? |
| "Download related papers" | Downloads the papers in the list | Should it also *find* more related papers? That's a second retrieval step |
| Hosting | Local | Chainlit needs a server host (Render / Heroku via Student Pack), not Streamlit Cloud. `history.db` → free hosted Postgres (Neon / Supabase). If the RAG uses torch + a local model, check the RAM limit (often 512 MB free) |
| Qdrant | — | Local folder mode allows **one process** at a time. For hosting, use Qdrant Cloud (free tier) |
