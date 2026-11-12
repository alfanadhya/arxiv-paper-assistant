"""Run the evaluation cases through the real search pipeline and save one trace per case (EVALUATION_PROTOCOL.md §2, §7.2).

Needs the full app environment: the Qdrant server with sebun_bge768_hybrid_full, Ollama with OLLAMA_MODEL,
and PyTorch (embedding model + reranker). Scoring (score.py) does NOT need any of these.

Run from the arxiv-paper-assistant folder, with the app's .env in place:
    .venv\\Scripts\\python.exe -X utf8 eval/run_eval.py --split dev --limit 2      # smoke test, 2 cases
    .venv\\Scripts\\python.exe -X utf8 eval/run_eval.py --split dev                # all dev cases
    .venv\\Scripts\\python.exe -X utf8 eval/run_eval.py --split test --final       # the ONE test run (§9)

Output, in a NEW folder eval/runs/<run_id>/ (an old run is never overwritten, §47):
    system_version.json   git commit, models, collection, settings, prompt hashes (§7)
    traces.jsonl          one line per case: parse, every Qdrant list, the rank trace, what was shown (§7.2),
                          and for 2-turn cases the analyze= answer
Each case is sent EXACTLY as written in the case file (request.setup_turns, then request.query), in one
simulated chat that keeps the GUI's state between turns; "{shown_labels}" becomes the last result's labels.
    run_log.txt           console output of the run
If a run stops, rerun with --run-id <same id>: cases already in traces.jsonl are skipped.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
CASES = EVAL / "cases"
RUNS = EVAL / "runs"
RUNNER_VERSION = "run_eval v1"
VERSION_RE = re.compile(r"v\d+$")


def norm_id(arxiv_id: str) -> str:
    """arXiv ID without its version suffix (unit of evaluation, §4)."""
    return VERSION_RE.sub("", str(arxiv_id or "").strip())


def load_cases(split: str, only: list[str] | None) -> list[dict]:
    cases = [json.loads(l) for f in sorted(CASES.glob("*-v*.jsonl")) for l in f.open(encoding="utf-8") if l.strip()]
    if split != "all":
        cases = [c for c in cases if c["split"] == split]
    if only:
        wanted = set(only)
        cases = [c for c in cases if c["case_id"] in wanted]
    return cases


FILTER_KEY_RE = re.compile(r"\s+(?:year|author)\s*=.*?(?=\s+\w+\s*=|$)", re.IGNORECASE)
LABELS_PLACEHOLDER = "{shown_labels}"


def build_messages(case: dict, variant: str) -> list[str]:
    """The messages typed in one chat, in order: setup turns, then the query, EXACTLY as in the case file.
    'plain' (an experiment) removes the year=/author= keys, to measure requests typed without them."""
    messages = list(case["request"]["setup_turns"]) + [case["request"]["query"]]
    if variant == "plain":
        messages = [FILTER_KEY_RE.sub("", m) for m in messages]
    return messages


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def ollama_model_info(app) -> dict:
    try:
        with urllib.request.urlopen(f"{app.OLLAMA_URL}/api/tags", timeout=5) as r:
            tags = json.load(r)["models"]
        m = next((t for t in tags if t["name"] == app.OLLAMA_MODEL), None)
        return {"digest": m["digest"], "quantization": m["details"].get("quantization_level"),
                "parameter_size": m["details"].get("parameter_size")} if m else {"error": "model not listed"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def system_version(app, retriever, variant: str) -> dict:
    """Everything that defines 'the system' for this run (§3 system version)."""
    import retriever_hybrid_rrf as rr
    import torch

    info = retriever.client.get_collection(app.COLLECTION)
    get_chains_src = ""
    try:
        import inspect
        get_chains_src = "".join(inspect.getsource(f) for f in (app.get_chains, getattr(app, "spell_check_chain", None))
                                 if f is not None)
    except Exception:
        pass
    prompts = {name: sha(value if isinstance(value, str) else repr(value.messages))
               for name, value in vars(app).items()
               if name.endswith("PROMPT") and (isinstance(value, str) or hasattr(value, "messages"))}
    return {
        "runner_version": RUNNER_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": [l for l in git("status", "--porcelain").splitlines() if l.strip()],
        "code_hashes": {f: sha((ROOT / f).read_text(encoding="utf-8"))
                        for f in ("threeofus_main.py", "retriever_hybrid_rrf.py", "prompt_cleaning.py",
                                  "chatbot_diagnostics.py")},
        "input_variant": variant,
        "qdrant": {"url": app.QDRANT_URL, "collection": app.COLLECTION, "points_count": info.points_count,
                   "indexed_vectors_count": info.indexed_vectors_count, "status": str(info.status)},
        "embedding_model": {"id": rr.MODEL_ID, "revision": rr.MODEL_REVISION, "query_instruction": rr.QUERY_INSTRUCTION},
        "sparse_model": {"id": rr.SPARSE_MODEL_ID, "min_weight": rr.MIN_SPARSE_WEIGHT, "max_terms": rr.MAX_SPARSE_TERMS},
        "fusion": "Qdrant server-side RRF over dense + sparse prefetch",
        "reranker_model": app.RERANKER_MODEL,
        "llm": {"model": app.OLLAMA_MODEL, "url": app.OLLAMA_URL, "num_ctx": app.NUM_CTX,
                "ollama": ollama_model_info(app),
                "temperatures_from_source": re.findall(r"temperature=([\d.]+)", get_chains_src),
                "seeds_from_source": re.findall(r"seed=(\d+)", get_chains_src)},
        "k": {"k_retrieve_per_search": app.K_RETRIEVE, "searches": ["topic", "survey of <topic>"],
              "papers_picked_by_llm": app.MAX_PAPERS, "surveys_shown": app.SURVEYS_SHOWN,
              "llm_candidates": f"N+{app.MAX_EXTRA_CANDIDATES} if it fits, else N+{app.MIN_EXTRA_CANDIDATES}",
              "ranking_output_reserve": app.RANKING_OUTPUT_RESERVE},
        "evidence_selection": "papers: reranker top N+1/N+2 sent to LLM, LLM picks up to 3; "
                              "surveys: reranker top 5 shown without LLM",
        "prompt_hashes": prompts,
        "machine": {"platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
                    "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }


def hit_record(h, rank: int) -> dict:
    """One Qdrant hit with the fields the scorer needs (relevance rules use title/year/authors/category)."""
    m = h.metadata or {}
    return {"rank": rank, "arxiv_id": norm_id(h.arxiv_id), "point_id": h.point_id, "rrf_score": round(h.vector_score, 5),
            "dense_rank": h.dense_rank, "sparse_rank": h.sparse_rank, "title": h.title, "year": m.get("year"),
            "authors": m.get("authors"), "author_keys": m.get("author_keys"),
            "author_last_names": m.get("author_last_names"), "primary_category": m.get("primary_category"),
            "categories": m.get("categories")}


def paper_record(h) -> dict:
    m = h.metadata or {}
    return {"arxiv_id": norm_id(h.arxiv_id), "title": h.title, "year": m.get("year"), "authors": m.get("authors"),
            "primary_category": m.get("primary_category"), "abstract": h.text.split("\n\n", 1)[-1]}


def stages(response, qdrant_calls: list) -> dict:
    """Each ranked list in pipeline order, as arXiv IDs (§7.2 trace.stages)."""
    cands = response.diagnostics.candidates
    by_pool = lambda pool: sorted((c for c in cands if c.pool == pool and c.rerank_rank),
                                  key=lambda c: c.rerank_rank)
    llm = sorted((c for c in cands if c.llm_rank and c.picked_by in ("llm", "fallback")), key=lambda c: c.llm_rank)
    shown = [response.papers[n - 1] for n in response.top_papers] + [response.surveys[n - 1] for n in response.top_surveys]
    last = max((c["attempt"] for c in qdrant_calls), default=0)
    return {
        "qdrant": {c["search"]: [h["arxiv_id"] for h in c["hits"]] for c in qdrant_calls if c["attempt"] == last},
        "rerank_papers": [norm_id(c.arxiv_id) for c in by_pool("paper")],
        "rerank_surveys": [norm_id(c.arxiv_id) for c in by_pool("survey")],
        "sent_to_llm": [norm_id(h.arxiv_id) for h in response.papers],
        "llm_ranked": [norm_id(c.arxiv_id) for c in llm],
        "shown_papers": [norm_id(response.papers[n - 1].arxiv_id) for n in response.top_papers],
        "shown_surveys": [norm_id(response.surveys[n - 1].arxiv_id) for n in response.top_surveys],
        "shown": [norm_id(h.arxiv_id) for h in shown],
    }


class Chat:
    """The per-chat state the GUI keeps between turns (threeofus_main.gui_on_message): remembered mode,
    conversation memory, the last result's labelled papers (for analyze=) and every paper shown so far."""

    def __init__(self, app, gui):
        self.app, self.gui = app, gui
        self.mode, self.history, self.shown, self.known = None, [], [], {}

    def labels(self) -> str:
        return ", ".join(p["label"] for p in self.shown)

    def send(self, message: str):
        message = message.replace(LABELS_PLACEHOLDER, self.labels())
        r = self.app.run_turn(message, log=lambda msg: None, history=list(self.history), current_mode=self.mode,
                              shown=list(self.shown), known=dict(self.known))
        self.mode = r.active_mode
        tags = self.app.memory_tags(r)
        self.history += [{"role": "user", "content": message, "tags": tags},
                         {"role": "assistant", "content": self.app.assistant_turn_text(r), "tags": tags}]
        self.history = self.history[-2 * self.app.MAX_HISTORY_TURNS:]
        if r.mode in ("search", "title") and not r.errors:
            best = r.best[0] if r.best else None
            self.shown = []
            for label, hit in self.gui.picked_hits(r):
                p = self.gui.to_paper(hit, label)
                self.known[p["arxiv_id"]] = p
                self.shown.append({**p, "best": label == best})
        return message, r


def run_case(app, gui, case: dict, variant: str, qdrant_log: list) -> dict:
    messages = build_messages(case, variant)
    chat = Chat(app, gui)
    turns, search_response, search_calls = [], None, []
    started = time.perf_counter()
    try:
        for message in messages:
            qdrant_log.clear()
            t0 = time.perf_counter()
            sent, r = chat.send(message)
            turns.append({"message_sent": sent, "mode": r.mode, "errors": list(r.errors), "answer": r.answer,
                          "papers_used": list(r.memory_papers), "tokens": r.diagnostics.context_usage,
                          "diagnostics": r.diagnostics.messages, "latency_s": round(time.perf_counter() - t0, 2)})
            if r.mode in ("search", "title"):
                search_response, search_calls = r, list(qdrant_log)
    except Exception as exc:   # one broken case must not stop the run; it is scored as a system error
        return {"case_id": case["case_id"], "messages": messages, "message_sent": messages[-1], "crashed": True,
                "turns": turns, "errors": [f"{type(exc).__name__}: {exc}"], "traceback": traceback.format_exc(),
                "latency_s": round(time.perf_counter() - started, 2)}
    latency = round(time.perf_counter() - started, 2)
    last = turns[-1]
    answer = None if last["mode"] in ("search", "title") else {
        "mode": last["mode"], "message_sent": last["message_sent"], "text": last["answer"],
        "papers_used": last["papers_used"], "errors": last["errors"]}
    if search_response is None:   # no search turn ran (e.g. the message went to chat mode): nothing to score
        return {"case_id": case["case_id"], "messages": messages, "message_sent": last["message_sent"],
                "crashed": False, "mode": last["mode"], "parse": None, "warnings": [],
                "errors": last["errors"] or [f"no search turn: the app handled the message in {last['mode']} mode"],
                "qdrant_calls": [], "stages": {"qdrant": {}, "rerank_papers": [], "rerank_surveys": [],
                                               "sent_to_llm": [], "llm_ranked": [], "shown_papers": [],
                                               "shown_surveys": [], "shown": []},
                "candidates": [], "gate": {"answered": False}, "context_ids": [], "context_papers": [],
                "response": {"shown": [], "best": None, "why_best": "", "criteria": [], "ranking_fallback": False,
                             "reasoning": "", "formatted": ""},
                "answer": answer, "turns": turns, "diagnostics": [], "tokens": [], "latency_s": latency}
    response, qdrant_calls = search_response, search_calls
    message = next(t["message_sent"] for t in reversed(turns) if t["mode"] in ("search", "title"))
    a = response.analysis
    shown_hits = [response.papers[n - 1] for n in response.top_papers] + [response.surveys[n - 1] for n in response.top_surveys]
    # search_papers() makes 2 calls (topic, survey); an author retry makes 2 more. The last attempt is the one used.
    calls = [dict(call, search=("topic", "survey")[i % 2], attempt=i // 2 + 1) for i, call in enumerate(qdrant_calls)]
    return {
        "case_id": case["case_id"],
        "messages": [t["message_sent"] for t in turns],   # every message typed, as sent (labels filled in)
        "message_sent": message,                          # the search=/title= message the stages below come from
        "crashed": False,
        "mode": response.mode,
        "parse": None if a is None else {
            "topic": a.corrected_query, "year": a.year, "year_from": a.year_from, "year_to": a.year_to,
            "authors": list(a.authors or []), "category": getattr(a, "category", None),
            "survey_focus": response.survey_focus},
        "warnings": list(response.warnings),
        "errors": list(response.errors),
        "qdrant_calls": calls,
        "stages": stages(response, calls),
        "candidates": [dict(asdict(c), arxiv_id=norm_id(c.arxiv_id)) for c in response.diagnostics.candidates],
        "gate": {"answered": bool(shown_hits) and not response.errors,
                 "rule": "no explicit gate: the system abstains only when it returns an error / no papers"},
        "context_ids": [norm_id(h.arxiv_id) for h in response.papers],
        "context_papers": [paper_record(h) for h in response.papers],
        "response": {"shown": [paper_record(h) for h in shown_hits], "best": response.best[0] if response.best else None,
                     "why_best": response.why_best, "criteria": list(response.criteria),
                     "ranking_fallback": response.ranking_fallback, "reasoning": response.reasoning,
                     "formatted": app.format_response(response)},
        "answer": answer,             # the analyze= turn of a 2-turn case (text + papers used), else None
        "turns": turns,
        "diagnostics": response.diagnostics.messages,
        "tokens": [u for t in turns for u in t["tokens"]],
        "latency_s": latency,
    }


class Tee:
    """Write console output to the run log as well."""
    def __init__(self, path: Path):
        self.file, self.stdout = path.open("a", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], required=True)
    ap.add_argument("--cases", nargs="*", help="only these case IDs, e.g. m00 ls001")
    ap.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    ap.add_argument("--input", choices=["keyed", "plain"], default="keyed",
                    help="keyed (default): send each message exactly as in the case file; "
                         "plain (experiment): remove the year=/author= keys first")
    ap.add_argument("--run-id", help="continue this run folder instead of starting a new one")
    ap.add_argument("--final", action="store_true", help="required for --split test/all: the test set is run ONCE (§9)")
    args = ap.parse_args()

    if args.split in ("test", "all") and not args.final and not args.limit:
        raise SystemExit("The test split is run once, on the frozen configuration (EVALUATION_PROTOCOL.md §9).\n"
                         "Add --final if this is that run, or use --split dev.")

    cases = load_cases(args.split, args.cases)[:args.limit] if args.limit else load_cases(args.split, args.cases)
    if not cases:
        raise SystemExit("No cases selected.")

    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M}_{args.split}_{args.input}" + ("_smoke" if args.limit else "")
    out = RUNS / run_id
    if out.exists() and not args.run_id:
        raise SystemExit(f"{out} already exists; never overwrite a run (§47). Use --run-id to continue it.")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "run_log.txt")

    os.chdir(ROOT)                       # .env, models_cache and .chainlit are found relative to the app folder
    sys.path.insert(0, str(ROOT))
    print(f"Run {run_id}: {len(cases)} cases ({args.split}, input={args.input})")
    print("Loading the app, models and Qdrant connection...")
    import threeofus_main as app
    import app as gui                    # the GUI's own helpers: labels of shown papers, paper dicts for analyze=

    count = app.warmup(print)
    retriever = app.get_retriever()
    qdrant_log: list = []
    original_retrieve = retriever.retrieve

    def logged_retrieve(query, k, metadata):   # keep every raw Qdrant list for R2 and the stage trace
        hits = original_retrieve(query, k, metadata)
        qdrant_log.append({"query": query, "k": k, "filters": repr(metadata),
                           "hits": [hit_record(h, i) for i, h in enumerate(hits, start=1)]})
        return hits

    retriever.retrieve = logged_retrieve
    print(f"Collection {app.COLLECTION}: {count:,} points" if count else "Collection point count unknown")

    version_file = out / "system_version.json"
    version = system_version(app, retriever, args.input)
    if version_file.exists():
        old = json.loads(version_file.read_text(encoding="utf-8"))
        changed = [k for k in ("git_commit", "code_hashes", "qdrant", "llm", "input_variant") if old.get(k) != version.get(k)]
        if changed:
            raise SystemExit(f"The system changed since this run started ({changed}); start a new run instead.")
    else:
        version_file.write_text(json.dumps(version, indent=2, ensure_ascii=False), encoding="utf-8")
    if version["git_dirty_files"]:
        print(f"WARNING: uncommitted changes {version['git_dirty_files']}; the git commit alone won't identify this system.")

    traces = out / "traces.jsonl"
    done = {json.loads(l)["case_id"] for l in traces.open(encoding="utf-8")} if traces.exists() else set()
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"{len(done)} cases already done, {len(todo)} to run.\n")

    started = time.perf_counter()
    for i, case in enumerate(todo, start=1):
        trace = run_case(app, gui, case, args.input, qdrant_log)
        trace.update(run_id=run_id, system_version_ref="system_version.json", split=case["split"],
                     case_set_version=case["case_set_version"])
        with traces.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        shown = trace.get("stages", {}).get("shown", [])
        status = "CRASH" if trace["crashed"] else ("error: " + trace["errors"][0][:60] if trace["errors"] else f"{len(shown)} shown")
        print(f"[{i}/{len(todo)}] {case['case_id']:<6} {trace['latency_s']:>6.1f}s  {status}")

    print(f"\nDone in {time.perf_counter() - started:.0f}s. Traces: {traces}")
    print(f"Score it with:  python -X utf8 eval/score.py {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
