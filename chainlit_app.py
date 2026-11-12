"""Chainlit chat window for the arXiv collection. Each message you type is a search; the answer is the
list of closest papers (title, link, year, category, authors, short abstract). No language model is
used, so nothing is generated: every line shown comes from the collection.

Put this file in the folder of retrieval.py and start it:
    pip install chainlit qdrant-client sentence-transformers
    chainlit run chainlit_app.py -w            # opens http://localhost:8000

Settings (environment variables, all optional):
    QDRANT_URL         Qdrant server (default http://localhost:6333); QDRANT_API_KEY is read by retrieval.py
    QDRANT_PATH        a local embedded Qdrant folder instead of a server (leave QDRANT_URL empty to use it)
    QDRANT_COLLECTION  collection name (default sebun_bge384_persistent_full)

In the chat window the gear icon opens the filters: category, year range, cs-primary only, number of
results, reranking by query-term coverage, and the abstain threshold. They are checked by retrieval.py
(MetadataIntent), the same way as the command-line tool.
"""
from __future__ import annotations

import os
import threading
from datetime import date

import chainlit as cl
from chainlit.input_widget import Slider, Switch, TextInput

import retrieval as R

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.environ.get("QDRANT_PATH") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION", "sebun_bge384_persistent_full")

YEAR_MIN, YEAR_MAX = R.MIN_YEAR, date.today().year
K_RETRIEVE = 30            # candidates fetched before reranking and cutting to the number shown
SNIPPET_CHARS = 350

DEFAULTS = {
    "category": "",
    "year_from": YEAR_MIN,
    "year_to": YEAR_MAX,
    "cs_primary_only": False,
    "k_final": 5,
    "rerank": True,
    "gate": 0.0,
}

_retriever = None
_lock = threading.Lock()


# --------------------------------------------------------------------------- logic (no Chainlit calls)
def get_retriever():
    """Build the Qdrant retriever and the embedding model once, on first use."""
    global _retriever
    with _lock:
        if _retriever is None:
            if QDRANT_PATH and not os.environ.get("QDRANT_URL"):
                _retriever = R.build_retriever(qdrant_path=QDRANT_PATH, collection=COLLECTION)
            else:
                _retriever = R.build_retriever(qdrant_url=QDRANT_URL, collection=COLLECTION)
        return _retriever


def settings_to_filters(settings: dict) -> dict:
    """Turn the settings panel values into the raw filter object retrieval.MetadataIntent.parse checks."""
    raw: dict = {}
    category = str(settings.get("category") or "").strip()
    if category:
        raw["category"] = category
    year_from, year_to = int(settings.get("year_from", YEAR_MIN)), int(settings.get("year_to", YEAR_MAX))
    if year_from > YEAR_MIN:
        raw["year_from"] = year_from
    if year_to < YEAR_MAX:
        raw["year_to"] = year_to
    if settings.get("cs_primary_only"):
        raw["cs_primary_only"] = True
    return raw


def run_search(retriever, query: str, settings: dict) -> dict:
    """Validate the filters and search; raises ValueError for a bad filter."""
    metadata = R.MetadataIntent.parse(settings_to_filters(settings))
    k_final = max(1, min(int(settings.get("k_final", 5)), K_RETRIEVE))
    gate = float(settings.get("gate") or 0.0)
    return R.search(retriever, query, metadata, K_RETRIEVE, k_final,
                    rerank=bool(settings.get("rerank", True)), threshold=gate if gate > 0 else None)


def format_results(report: dict) -> str:
    """Markdown for the chat message."""
    intent = {k: v for k, v in report["metadata_intent"].items() if v not in (None, False)}
    filters = ", ".join(f"{k}={v}" for k, v in intent.items()) or "none"
    gate, items = report["gate"], report["candidates"]
    head = f"Filters: {filters} · reranking: {'on' if report['reranker'] else 'off'}"
    if not items:
        return head + "\n\nNo papers match these filters."
    if gate is not None and not gate["passed"]:
        return (head + f"\n\n**No close match.** The best result covers too few of your query words "
                f"({gate['reason']}). Try different words, or lower the abstain threshold in the settings.")
    lines = [head, ""]
    for item in items:
        meta = item["metadata"]
        body = " ".join(item["text"].split("\n\n", 1)[-1].split())
        if len(body) > SNIPPET_CHARS:
            body = body[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."
        info = " · ".join(str(x) for x in (meta.get("year"), meta.get("primary_category"),
                                           f"arXiv:{item['arxiv_id']}") if x)
        lines.append(f"**{item['final_rank']}. [{item['title']}]({item['url']})**  \n"
                     f"{info} · similarity {item['vector_score']:.3f} · query-word coverage "
                     f"{item['coverage_score']:.0%}")
        authors = R.format_authors(meta)
        if authors:
            lines.append(f"{authors}")
        lines.append(f"> {body}\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- Chainlit handlers
@cl.on_chat_start
async def start():
    cl.user_session.set("settings", dict(DEFAULTS))
    await cl.ChatSettings([
        TextInput(id="category", label="Category (e.g. cs.CL, cs.CV, stat.ML)", initial="",
                  placeholder="any"),
        Slider(id="year_from", label="From year", initial=YEAR_MIN, min=YEAR_MIN, max=YEAR_MAX, step=1),
        Slider(id="year_to", label="To year", initial=YEAR_MAX, min=YEAR_MIN, max=YEAR_MAX, step=1),
        Switch(id="cs_primary_only", label="Only papers whose main category is computer science",
               initial=False),
        Slider(id="k_final", label="Number of results", initial=5, min=1, max=15, step=1),
        Switch(id="rerank", label="Rerank by query-word coverage", initial=True),
        Slider(id="gate", label="Abstain below this query-word coverage (0 = never)", initial=0.0,
               min=0.0, max=1.0, step=0.05),
    ]).send()
    notice = cl.Message(content="Connecting to the paper collection (the first start loads the embedding model)...")
    await notice.send()
    try:
        retriever = await cl.make_async(get_retriever)()
    except SystemExit as exc:                 # retrieval.py reports its setup problems this way
        notice.content = f"Could not open the collection:\n\n```\n{exc}\n```"
        await notice.update()
        return
    except Exception as exc:                  # server down, model download failed, ...
        notice.content = f"Could not open the collection: {exc}"
        await notice.update()
        return
    count = f"{retriever.points_count:,}" if getattr(retriever, "points_count", None) else "many"
    notice.content = (f"Ready: {count} arXiv papers in `{COLLECTION}`. Type what you are looking for, "
                      "for example *graph neural networks for molecules*. The gear icon sets filters.")
    await notice.update()


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("settings", {**DEFAULTS, **settings})


@cl.on_message
async def on_message(message: cl.Message):
    query = (message.content or "").strip()
    if not query:
        return
    settings = cl.user_session.get("settings") or dict(DEFAULTS)
    try:
        retriever = await cl.make_async(get_retriever)()
        report = await cl.make_async(run_search)(retriever, query, settings)
    except ValueError as exc:                 # bad filter, e.g. a category written wrongly
        await cl.Message(content=f"Check the filters: {exc}").send()
        return
    except SystemExit as exc:
        await cl.Message(content=f"Could not open the collection:\n\n```\n{exc}\n```").send()
        return
    except Exception as exc:
        await cl.Message(content=f"The search failed: {exc}").send()
        return
    await cl.Message(content=format_results(report)).send()
