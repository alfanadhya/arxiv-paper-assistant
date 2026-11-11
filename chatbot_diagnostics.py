"""Diagnostic digest for the arXiv chatbot (threeofus_main.py).

Holds the query at each stage (original / follow-up rewrite / modified query sent to Qdrant) as separate
fields, tagged notes such as [mode], [analysis], [validation], [memory], [rerank], [ranking], [selection],
and a rank trace that follows every candidate from Qdrant through the reranker and the LLM to the display.
Self-contained: does not import threeofus_main, so it can be imported from the GUI and the CLI alike.
"""

import contextvars
from dataclasses import asdict, dataclass, field

from langchain_core.callbacks import BaseCallbackHandler

TAGS = ("[query]", "[mode]", "[analysis]", "[validation]", "[memory]", "[rerank]", "[ranking]", "[selection]",
        "[context]")
CONTEXT_WARN_PCT = 90   # warn when a call used at least this share of the model's context window
TITLE_TRACE_ROWS = 5    # title lookups: rows shown in the rank trace (top contenders)

# List that LLM calls of the current query report their token usage into (set per query, per thread).
_usage_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar("context_usage_sink", default=None)


# LangChain callback: reads the token counts Ollama already returns with each reply (no extra cost).
class ContextUsageHandler(BaseCallbackHandler):
    # Remember which pipeline stage this call belongs to and where to report.
    def __init__(self, stage: str, sink: list):
        self.stage, self.sink = stage, sink

    # Called when the model finishes; records prompt/output tokens and why generation stopped.
    def on_llm_end(self, response, **kwargs) -> None:
        for generations in response.generations:
            for g in generations:
                msg = getattr(g, "message", None)
                meta = getattr(msg, "response_metadata", None) or {}
                usage = getattr(msg, "usage_metadata", None) or {}
                prompt = meta.get("prompt_eval_count", usage.get("input_tokens"))
                output = meta.get("eval_count", usage.get("output_tokens"))
                if prompt is None and output is None:
                    continue
                self.sink.append({"stage": self.stage, "prompt_tokens": int(prompt or 0),
                                  "output_tokens": int(output or 0), "done_reason": meta.get("done_reason")})


# Start collecting token usage for this query into diag.context_usage.
def start_context_tracking(diag: "Diagnostics") -> None:
    _usage_sink.set(diag.context_usage)


# Stop collecting (so later, untracked calls in the same worker thread don't report into an old query).
def stop_context_tracking() -> None:
    _usage_sink.set(None)


# LangChain invoke config that reports this call's token usage (empty config if tracking is off).
def llm_config(stage: str) -> dict:
    sink = _usage_sink.get()
    return {"callbacks": [ContextUsageHandler(stage, sink)]} if sink is not None else {}


# One retrieved candidate and its rank at each stage of the pipeline (None = stage not reached).
@dataclass
class CandidateRank:
    chunk_id: str
    arxiv_id: str
    title: str
    pool: str                          # "paper" or "survey"
    search: str                        # Qdrant search that found it first: "topic" or "survey"
    search_rank: int                   # position in that Qdrant result list (hybrid RRF order)
    rrf_score: float                   # Qdrant hybrid (RRF) score
    dense_rank: int | None = None      # rank in the dense (BGE) list inside Qdrant's fusion
    sparse_rank: int | None = None     # rank in the sparse (BM25) list inside Qdrant's fusion
    initial_rank: int = 0              # position in its pool before reranking
    rerank_rank: int | None = None     # position after the cross-encoder reranker
    rerank_score: float | None = None  # cross-encoder relevance score (higher = better)
    llm_label: str | None = None       # label shown to the LLM ("2", "S1") if it was sent
    llm_rank: int | None = None        # order of the LLM's pick (1 = its best)
    final_rank: int | None = None      # position in the displayed list
    picked_by: str | None = None       # "llm" or "fallback"
    year: int | None = None            # publication year (from the Qdrant metadata already returned)
    title_sim: float | None = None     # title similarity to the typed title (title lookups only)


# Publication year from a hit's metadata, or None.
def hit_year(hit) -> int | None:
    year = (getattr(hit, "metadata", None) or {}).get("year")
    return year if isinstance(year, int) else None


# Diagnostic buffer: queries at each stage, tagged messages and the candidate rank trace.
@dataclass
class Diagnostics:
    original_prompt: str = ""                 # search= / title= text as typed by the user
    rewritten_prompt: str | None = None       # after the follow-up rewrite (None if not rewritten)
    modified_query: str | None = None         # final topic sent to Qdrant (after spelling/filler cleanup)
    mode: str = "search"                      # "search" (database search) or "chat" (direct LLM answer)
    messages: list[str] = field(default_factory=list)          # tagged lines
    candidates: list[CandidateRank] = field(default_factory=list)   # rank trace
    context_usage: list[dict] = field(default_factory=list)    # token usage per LLM call (stage, prompt, output)

    # Diagnostics for a direct-chat message (no database search).
    @classmethod
    def for_chat(cls, message: str) -> "Diagnostics":
        d = cls(original_prompt=message, mode="chat")
        d.append("[mode] Chat with the LLM, no database search "
                 "(add search=<topic> or title=<paper title> to search the papers).")
        return d

    # Wrap up a search: the mode key (search= / title=) and the parser's notes first, context usage last.
    def finish_search(self, query: str, num_ctx: int, keyword: str = "search",
                      parser_notes: list[str] | None = None) -> None:
        what = "Exact-title lookup" if keyword == "title" else "Search"
        self.messages[:0] = [f"[mode] {what} requested with {keyword}= -> '{query}'."] + list(parser_notes or [])
        self.messages.extend(self.context_messages(num_ctx))

    # Add one message.
    def append(self, message: str) -> None:
        self.messages.append(message)

    # Add several messages with +=.
    def __iadd__(self, messages: list[str]) -> "Diagnostics":
        self.messages.extend(messages)
        return self

    # Iterate over the messages.
    def __iter__(self):
        return iter(self.messages)

    # Number of messages.
    def __len__(self) -> int:
        return len(self.messages)

    # True if the searched query differs from what the user typed (rewrite or spelling/filler fix).
    @property
    def query_changed(self) -> bool:
        norm = lambda s: " ".join((s or "").lower().split())
        return bool(self.rewritten_prompt) or (
            bool(self.modified_query) and norm(self.modified_query) != norm(self.original_prompt))

    # ------------------------------------------------------------------ rank trace
    # Record a pool of retrieved hits in their initial order; origin maps chunk id -> (search, rank).
    def track_pool(self, pool: str, hits: list, origin: dict) -> None:
        for i, h in enumerate(hits, start=1):
            search, search_rank = origin.get(h.point_id, ("topic", getattr(h, "initial_rank", i)))
            self.candidates.append(CandidateRank(
                chunk_id=h.point_id, arxiv_id=h.arxiv_id, title=h.title, pool=pool, search=search,
                search_rank=search_rank, rrf_score=round(float(getattr(h, "vector_score", 0.0)), 4),
                dense_rank=getattr(h, "dense_rank", None), sparse_rank=getattr(h, "sparse_rank", None),
                initial_rank=i, year=hit_year(h)))

    # Candidate record for a chunk id (None if not tracked).
    def candidate(self, chunk_id: str) -> CandidateRank | None:
        return next((c for c in self.candidates if c.chunk_id == chunk_id), None)

    # Record the reranker order and scores for hits (already sorted best first).
    def track_rerank(self, hits: list, scores: list[float]) -> None:
        for i, (h, s) in enumerate(zip(hits, scores), start=1):
            c = self.candidate(h.point_id)
            if c:
                c.rerank_rank, c.rerank_score = i, round(float(s), 4)

    # Record title similarities (title lookups): chunk id -> similarity in [0, 1].
    def track_title_sims(self, sims: dict) -> None:
        for c in self.candidates:
            if c.chunk_id in sims:
                c.title_sim = round(float(sims[c.chunk_id]), 4)

    # Record which hits were sent to the LLM and under which label prefix ("" or "S").
    def track_sent(self, hits: list, prefix: str = "") -> None:
        for i, h in enumerate(hits, start=1):
            c = self.candidate(h.point_id)
            if c:
                c.llm_label = f"{prefix}{i}"

    # Record a reranked pool: ranks/scores, and a [rerank] note on how many were promoted into the top `keep`.
    def note_rerank(self, pool_name: str, original: list, ranked: list, scores: list[float], keep: int,
                    model: str) -> None:
        self.track_rerank(ranked, scores)
        kept_ids = {h.point_id for h in ranked[:keep]}
        promoted = sum(1 for i, h in enumerate(original) if i >= keep and h.point_id in kept_ids)
        self.append(f"[rerank] {len(original)} {pool_name}(s) reranked with {model}; top {keep} sent to the LLM "
                    f"({promoted} promoted from below Qdrant's top {keep}, {promoted} dropped).")

    # Record picks from 1-based numbers: paper_order/survey_order in the LLM's order, top_* as displayed.
    def record_picks(self, papers: list, surveys: list, paper_order: list[int], survey_order: list[int],
                     top_papers: list[int], top_surveys: list[int], picked_by: str,
                     survey_picked_by: str | None = None) -> None:
        # Papers and surveys are ranked by separate LLM calls: each pool's pick rank starts at 1 and
        # records its own picked_by ("llm" or "fallback"), since one pool can fall back alone.
        survey_picked_by = survey_picked_by or picked_by
        shown = [papers[n - 1] for n in top_papers] + [surveys[n - 1] for n in top_surveys]
        self.track_picks([papers[n - 1] for n in paper_order], [], picked_by)
        self.track_picks([surveys[n - 1] for n in survey_order], shown, survey_picked_by)

    # Record picks: llm_order = hits in the LLM's order, final_order = hits as displayed.
    def track_picks(self, llm_order: list, final_order: list, picked_by: str) -> None:
        for i, h in enumerate(llm_order, start=1):
            c = self.candidate(h.point_id)
            if c:
                c.llm_rank, c.picked_by = i, picked_by
        for i, h in enumerate(final_order, start=1):
            c = self.candidate(h.point_id)
            if c:
                c.final_rank = i

    # Candidates worth showing: everything sent to the LLM, plus initial top ones the reranker dropped.
    def rank_rows(self) -> list[CandidateRank]:
        if self.mode == "title":   # title lookups: the top contenders after title similarity + reranker
            rows = [c for c in self.candidates if c.rerank_rank and c.rerank_rank <= TITLE_TRACE_ROWS]
            return sorted(rows, key=lambda c: c.rerank_rank)
        rows = []
        for pool in ("paper", "survey"):
            items = [c for c in self.candidates if c.pool == pool]
            sent = sum(1 for c in items if c.llm_label)
            rows += [c for c in items if c.llm_label or (sent and c.initial_rank <= sent)]
        key = lambda c: (c.pool != "paper", c.final_rank or 99, c.llm_rank or 99, c.rerank_rank or 99)
        return sorted(rows, key=key)

    # One line per shown candidate: Qdrant rank -> rerank -> LLM -> displayed.
    def rank_lines(self) -> list[str]:
        lines = []
        for c in self.rank_rows():
            rerank = f"rerank {c.rerank_rank} ({c.rerank_score:.3f})" if c.rerank_rank else "rerank -"
            sent = f"sent [{c.llm_label}]" if c.llm_label else "not sent"
            llm = f"{c.picked_by} pick {c.llm_rank}" if c.llm_rank else "not picked"
            shown = f"shown {c.final_rank}" if c.final_rank else "not shown"
            lines.append(f"[{c.pool}] qdrant {c.initial_rank} ({c.search} search #{c.search_rank}) -> {rerank} "
                         f"-> {sent} -> {llm} -> {shown} | {c.year or '----'} | "
                         + (f"title sim {c.title_sim:.2f} | " if c.title_sim is not None else "")
                         + f"{c.title[:70]}")
        return lines

    # ------------------------------------------------------------------ context usage
    # One [context] line per LLM call plus the peak, against the model's context window num_ctx.
    def context_messages(self, num_ctx: int) -> list[str]:
        if not self.context_usage:
            return ["[context] No LLM token counts were reported for this query."]
        lines, peak = [], None
        for u in self.context_usage:
            total = u["prompt_tokens"] + u["output_tokens"]
            pct = round(100 * total / num_ctx) if num_ctx else 0
            line = (f"[context] {u['stage']}: prompt {u['prompt_tokens']:,} + output {u['output_tokens']:,} "
                    f"= {total:,} / {num_ctx:,} tokens ({pct}%)")
            if u.get("done_reason") == "length":
                line += " WARNING: output hit the limit and was cut off"
            elif pct >= CONTEXT_WARN_PCT:
                line += " WARNING: close to the context limit, input may be cut"
            lines.append(line)
            if peak is None or pct > peak[0]:
                peak = (pct, u["stage"])
        lines.append(f"[context] Peak use {peak[0]}% ({peak[1]}) of the {num_ctx:,}-token context.")
        return lines

    # ------------------------------------------------------------------ output
    # Plain dict for the GUI panel (JSON-serializable).
    def to_dict(self) -> dict:
        return {"original_prompt": self.original_prompt, "rewritten_prompt": self.rewritten_prompt,
                "modified_query": self.modified_query, "mode": self.mode, "messages": list(self.messages),
                "ranks": [asdict(c) for c in self.rank_rows()], "context_usage": list(self.context_usage)}

    # Query fields, messages and rank trace as display lines.
    def as_lines(self) -> list[str]:
        lines = [f"[query] Mode: {self.mode}", f"[query] Original prompt: {self.original_prompt}"]
        if self.rewritten_prompt:
            lines.append(f"[query] Follow-up rewrite: {self.rewritten_prompt}")
        if self.mode == "search":
            lines.append(f"[query] Modified query (searched): {self.modified_query or '(none)'}")
        return lines + self.messages + self.rank_lines()


# Render the diagnostic digest as plain text; accepts a Diagnostics or anything with .diagnostics.
def format_diagnostics(source) -> str:
    d = source if isinstance(source, Diagnostics) else source.diagnostics
    lines = ["\n=== Diagnostic digest ===",
             f"  Mode            : {d.mode}",
             f"  Original prompt : {d.original_prompt}"]
    if d.rewritten_prompt:
        lines.append(f"  Follow-up rewrite: {d.rewritten_prompt}")
    if d.mode == "search":
        lines.append(f"  Modified query  : {d.modified_query or '(none)'}")
    lines += [f"  {m}" for m in d.messages] or ["  (no issues)"]
    rank_lines = d.rank_lines()
    if rank_lines:
        lines.append("  Rank trace (qdrant -> rerank -> LLM -> shown):")
        lines += [f"    {r}" for r in rank_lines]
    return "\n".join(lines)


# Short summary of what the query analyzer returned (any object with the QueryAnalysis fields).
def describe_analysis(a) -> str:
    return (f"topic='{a.corrected_query}', authors={a.authors}, year={a.year}, "
            f"year_from={a.year_from}, year_to={a.year_to}")
