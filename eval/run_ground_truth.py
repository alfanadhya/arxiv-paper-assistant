"""Send each paper of eval/data/ground_truth.jsonl to the app as a plain message (no search= / title= key), the
way a user types it in the GUI, and record what every stage of the pipeline returned.

The app decides the route itself (without a key: the agent, which runs its own searches). Every message is a
fresh chat: no history, no papers on screen, the default reranker cutoff.

Run from the project folder (Qdrant and Ollama must be up):
    python -X utf8 eval/run_ground_truth.py                    # the paper's title as the message
    python -X utf8 eval/run_ground_truth.py --input question   # the paper's general_question as the message
    python -X utf8 eval/run_ground_truth.py --key title        # sent as 'title=<title>' (the title route)
    python -X utf8 eval/run_ground_truth.py --limit 2          # first 2 papers only

Output, in a new folder eval/runs/<run_id>/:
    records.jsonl         one line per paper, written as soon as that paper is done
    system_version.json   git commit, models, collection, settings
    run_log.txt           console output
If a run stops, rerun with --run-id <same id>: papers already in records.jsonl are skipped.

One record (all id lists are in rank order, best first, arXiv ids without version):
    target_id, target_title    the paper the message is about (the one relevant paper)
    message_sent               the exact text sent
    mode                       route the app chose ("agent", "search", "title")
    modified_queries           the query the app searched Qdrant with, one per search
    searches                   one entry per search the app ran:
        query, authors, year_from, year_to   what was searched and with which filters
        qdrant_ids             Qdrant's hits (hybrid RRF order)
        reranked_ids           the hits after the reranker, with reranked_scores next to them
        sent_to_llm_ids        the reranked hits at or above the cutoff that were passed to the LLM
    stages                     the same lists for the whole message (searches merged, see merge_stages):
        qdrant, reranked, sent_to_llm, llm_selected (the papers the LLM cited / the app showed; on search= and
        title= also the surveys shown, which skip the LLM, so a shown survey has no sent_to_llm rank)
    target_rank                1-based position of the target in each of the four stage lists (null = absent)
    titles                     arXiv id -> title for every id in the record
    answer, warnings, errors, refused_out_of_scope, diagnostics, tokens, latency_s
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
sys.path.insert(0, str(EVAL))
from run_eval import RUNS, Tee, norm_id, system_version   # noqa: E402

GROUND_TRUTH = EVAL / "data" / "ground_truth.jsonl"


def first_seen(lists: list[list[str]]) -> list[str]:
    """Several ranked lists as one: round-robin by rank (all rank 1s, then all rank 2s, ...), duplicates dropped."""
    merged: list[str] = []
    for rank in range(max((len(l) for l in lists), default=0)):
        for l in lists:
            if rank < len(l) and l[rank] not in merged:
                merged.append(l[rank])
    return merged


def merge_stages(searches: list[dict], selected: list[str]) -> dict:
    """The stage lists for the whole message. With one search they are that search's lists. With several:
    qdrant and sent_to_llm are interleaved by rank (first_seen); reranked is ordered by each paper's best
    reranker score over the searches."""
    best: dict[str, float] = {}
    for s in searches:
        for pid, score in zip(s["reranked_ids"], s["reranked_scores"]):
            best[pid] = max(score, best.get(pid, float("-inf")))
    return {"qdrant": first_seen([s["qdrant_ids"] for s in searches]),
            "reranked": sorted(best, key=lambda pid: -best[pid]),
            "sent_to_llm": first_seen([s["sent_to_llm_ids"] for s in searches]),
            "llm_selected": selected}


def rank_of(target: str, ids: list[str]) -> int | None:
    return ids.index(target) + 1 if target in ids else None


def agent_searches(response, qdrant_log: list) -> list[dict]:
    """One entry per search_papers call of the agent, with the Qdrant list that call got. A search makes one
    Qdrant call; an author search with no hits makes a second one (last name only), which is the one used."""
    calls = list(qdrant_log)
    out = []
    for n, s in enumerate(response.agent_searches, start=1):
        call = calls.pop(0) if calls else {"hits": []}
        if not call["hits"] and s["authors"] and calls and s["last_name_only"] is not None and calls[0]["query"] == call.get("query"):
            call = calls.pop(0)
        out.append({"n": n, "query": s["query"], "authors": s["authors"], "year_from": s["year_from"],
                    "year_to": s["year_to"], "cutoff": s["cutoff"],
                    "qdrant_ids": [h["arxiv_id"] for h in call["hits"]],
                    "reranked_ids": [norm_id(r["arxiv_id"]) for r in s["ranked"]],
                    "reranked_scores": [r["rerank_score"] for r in s["ranked"]],
                    "sent_to_llm_ids": [norm_id(p) for p in s["returned"]]})
    return out


def pipeline_search(response, qdrant_log: list) -> list[dict]:
    """The same entry for the fixed routes (search= / title=), in case the app picks one: one search, read from
    the rank trace."""
    cands = response.diagnostics.candidates
    reranked = sorted((c for c in cands if c.rerank_rank), key=lambda c: c.rerank_rank)
    a = response.analysis
    return [{"n": 1, "query": a.corrected_query if a else "", "authors": list(a.authors or []) if a else [],
             "year_from": a.year_from if a else None, "year_to": a.year_to if a else None, "cutoff": None,
             "qdrant_ids": [h["arxiv_id"] for h in qdrant_log[-1]["hits"]] if qdrant_log else [],
             "reranked_ids": [norm_id(c.arxiv_id) for c in reranked],
             "reranked_scores": [c.rerank_score for c in reranked],
             "sent_to_llm_ids": [norm_id(h.arxiv_id) for h in response.papers]}]


def run_paper(app, paper: dict, message: str, qdrant_log: list) -> dict:
    target = norm_id(paper["arxiv_id"])
    record = {"target_id": target, "target_title": paper["title"], "message_sent": message}
    qdrant_log.clear()
    started = time.perf_counter()
    try:
        response = app.run_turn(message, log=lambda msg: None)
    except Exception as exc:   # one broken paper must not stop the run
        return {**record, "crashed": True, "errors": [f"{type(exc).__name__}: {exc}"],
                "traceback": traceback.format_exc(), "latency_s": round(time.perf_counter() - started, 2)}
    latency = round(time.perf_counter() - started, 2)
    searches = agent_searches(response, qdrant_log) if response.mode == "agent" else pipeline_search(response, qdrant_log)
    # shown = the papers the LLM picked, plus (search= / title=) the surveys shown next to them: surveys skip the
    # LLM (the reranker's top surveys are shown directly), so a target survey has llm_selected but no sent_to_llm
    selected = [norm_id(response.papers[n - 1].arxiv_id) for n in response.top_papers]
    selected += [norm_id(response.surveys[n - 1].arxiv_id) for n in getattr(response, "top_surveys", [])
                 if norm_id(response.surveys[n - 1].arxiv_id) not in selected]
    stages = merge_stages(searches, selected)
    titles = {h["arxiv_id"]: h["title"] for call in qdrant_log for h in call["hits"]}
    titles.update({norm_id(h.arxiv_id): h.title for h in response.papers})
    return {**record, "crashed": False, "mode": response.mode,
            "modified_queries": [s["query"] or "; ".join(s["authors"]) for s in searches],
            "searches": searches,
            "stages": stages,
            "target_rank": {stage: rank_of(target, ids) for stage, ids in stages.items()},
            "answer": response.answer,
            "warnings": list(response.warnings), "errors": list(response.errors),
            "refused_out_of_scope": app.SCOPE_REFUSAL in response.errors,
            "titles": titles,
            "diagnostics": response.diagnostics.messages,
            "tokens": response.diagnostics.context_usage,
            "latency_s": latency}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["title", "question"], default="title",
                    help="what is typed: the paper's title (default) or its general_question")
    ap.add_argument("--key", choices=["title", "search"],
                    help="put this mode key in front of the message, e.g. --key title sends 'title=<text>' "
                         "(default: no key, the app picks the route)")
    ap.add_argument("--limit", type=int, help="only the first N papers")
    ap.add_argument("--run-id", help="continue this run folder instead of starting a new one")
    ap.add_argument("--only", nargs="*", help="only these arXiv ids, e.g. 1606.05250 2106.09685")
    args = ap.parse_args()

    papers = [json.loads(l) for l in GROUND_TRUTH.open(encoding="utf-8") if l.strip()]
    if args.only:
        papers = [p for p in papers if norm_id(p["arxiv_id"]) in set(args.only)]
    papers = papers[:args.limit]
    field = "title" if args.input == "title" else "general_question"
    prefix = f"{args.key}=" if args.key else ""
    run_id = args.run_id or (f"{datetime.now():%Y%m%d-%H%M}_ground_truth_{args.input}"
                             + (f"_{args.key}_key" if args.key else ""))
    out = RUNS / run_id
    if out.exists() and not args.run_id:
        raise SystemExit(f"{out} already exists; use --run-id to continue it.")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "run_log.txt")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    print(f"Run {run_id}: {len(papers)} papers, message = the paper's {field}, "
          + (f"mode key '{prefix}'" if prefix else "no mode key"))
    print("Loading the app, models and Qdrant connection...")
    import threeofus_main as app

    app.warmup(print)
    retriever = app.get_retriever()
    qdrant_log: list = []
    original_retrieve = retriever.retrieve

    def logged_retrieve(query, k, metadata, sparse_query=None):   # keep every raw Qdrant list
        hits = original_retrieve(query, k, metadata, sparse_query=sparse_query)
        qdrant_log.append({"query": query, "sparse_query": sparse_query, "k": k, "filters": repr(metadata),
                           "hits": [{"arxiv_id": norm_id(h.arxiv_id), "title": h.title} for h in hits]})
        return hits

    retriever.retrieve = logged_retrieve

    version_file = out / "system_version.json"
    if not version_file.exists():
        version = dict(system_version(app, retriever, f"ground_truth_{args.input}", "none"),
                       runner_version="run_ground_truth v1", rerank_cutoff=app.RERANK_CUTOFF,
                       ground_truth_file=str(GROUND_TRUTH.relative_to(ROOT)))
        version_file.write_text(json.dumps(version, indent=2, ensure_ascii=False), encoding="utf-8")

    records = out / "records.jsonl"
    done = {json.loads(l)["target_id"] for l in records.open(encoding="utf-8")} if records.exists() else set()
    todo = [p for p in papers if norm_id(p["arxiv_id"]) not in done]
    print(f"{len(done)} papers already done, {len(todo)} to run. Records: {records}\n")

    started = time.perf_counter()
    for i, paper in enumerate(todo, start=1):
        record = run_paper(app, paper, prefix + " ".join(str(paper[field]).split()), qdrant_log)
        record.update(run_id=run_id, input=args.input)
        with records.open("a", encoding="utf-8") as f:   # written and closed per paper: readable while running
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if record["crashed"]:
            status = "CRASH " + record["errors"][0][:70]
        else:
            r = record["target_rank"]
            status = (f"{record['mode']}, {len(record['searches'])} search(es); target rank: qdrant {r['qdrant']}, "
                      f"reranked {r['reranked']}, sent {r['sent_to_llm']}, selected {r['llm_selected']}"
                      + (f"; error: {record['errors'][0][:50]}" if record["errors"] else ""))
        print(f"[{i}/{len(todo)}] {record['target_id']:<11} {record['latency_s']:>6.1f}s  {status}")

    print(f"\nDone in {time.perf_counter() - started:.0f}s. Records: {records}")


if __name__ == "__main__":
    main()
