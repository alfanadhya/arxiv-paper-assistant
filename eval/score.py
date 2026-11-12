"""Score a saved run: rule-based metrics, failure labels, report and audit sheet (EVALUATION_PROTOCOL.md §4, §7.3, §8, §10).

Needs only the standard library and the run folder (traces.jsonl + system_version.json), so it runs on any
PC, without Qdrant, Ollama or PyTorch.

    python -X utf8 eval/score.py eval/runs/<run_id>

Writes into the run folder (overwriting only its own earlier score files):
    per_case.jsonl   metrics, stage ranks, gate cell, parse check and failure label per case (§7.3)
    report.json      aggregates with k/n counts, exclusions, per-slice tables, Wilson intervals (§10)
    audit.csv        one row per case for the human audit: what was shown, plus empty columns to fill (§7.3, §35)

Metric definitions (evaluator version "score v1"; unit = arXiv paper without version suffix, counted once per list):
    R1 recall_shown       relevant papers among everything shown (<=3 papers + <=5 surveys) / all known relevant
    R1p recall_papers     same, research papers shown only (<=3)
    R2 candidate_recall   relevant papers in either Qdrant list (topic + survey search, 20 each) / all known relevant
    R3 precision_shown    relevant papers shown / papers shown (rule-based cases use the relevance rule)
    R4 rr                 1 / rank of the first relevant paper in the shown list; 0 if none
    R5 ndcg_at_10         reranked list (both pools merged by reranker score), binary gain, ideal over the judged pool
    R6 stage_ranks        rank of each relevant paper at each stage, and the stage where it was dropped
    R1-R5 are None when the case has no enumerated relevant set (rule-based or abstain cases); exclusions are counted.
    G1 gate cell          positive = sufficiency_label; answered = papers shown and no error
    Q1 filters_correct    year_from / year_to / author of the parse equal the expected parse (topic: human audit)
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL = Path(__file__).resolve().parent
CASES = EVAL / "cases"
EVALUATOR_VERSION = "score v1"
PROTOCOL = EVAL / "EVALUATION_PROTOCOL.md"
H1_SLICE = "ls_specific"
H_THRESHOLD = 0.60


# --------------------------------------------------------------------------- relevance
def name_parts(full: str) -> tuple[str, str]:
    """('Mohit Bansal') -> ('bansal', 'm'); also accepts 'Bansal, Mohit'."""
    full = full.strip()
    if "," in full:
        last, first = [p.strip() for p in full.split(",", 1)]
    else:
        parts = full.split()
        last, first = parts[-1], " ".join(parts[:-1])
    return last.lower(), (first[:1].lower() if first else "")


def rule_matches(rule: dict, paper: dict) -> bool:
    """Deterministic relevance rule (§6.3): title regex AND every other key given."""
    if not re.search(rule["title_regex"], paper.get("title") or "", re.IGNORECASE):
        return False
    if "year" in rule:
        y, want = paper.get("year"), rule["year"]
        if isinstance(want, dict):
            if y is None or not (want.get("from", -1e9) <= y <= want.get("to", 1e9)):
                return False
        elif y != want:
            return False
    if "primary_category" in rule and paper.get("primary_category") != rule["primary_category"]:
        return False
    if "author" in rule:
        last, initial = name_parts(rule["author"])
        keys = paper.get("author_keys") or []
        lasts = [l.lower() for l in paper.get("author_last_names") or []]
        if keys:
            if not any(k == f"{last}_{initial}" or (k.startswith(f"{last}_") and not initial) for k in keys):
                return False
        elif last not in lasts:
            return False
    return True


def is_survey(paper: dict) -> bool:
    return bool(re.search(r"\bsurveys?\b", paper.get("title") or "", re.IGNORECASE))


class Judge:
    """Binary relevance for one case (§6.3)."""

    def __init__(self, case: dict):
        ref = case["reference"]
        v2 = int(re.search(r"v(\d+)", case["case_set_version"]).group(1)) >= 2
        self.listed = set(ref["relevant_ids"]) | (set(ref["acceptable_alternative_ids"]) if v2 else set())
        self.wrong_scope = set(ref["wrong_scope_ids"])
        self.rule = ref["relevance_rule"]
        self.surveys_count = case["surveys_count"]
        # Recall-type metrics need the full relevant set: only defined when it is enumerated and there is no rule
        self.enumerated = bool(self.listed) and self.rule is None

    def relevant(self, paper: dict) -> bool:
        pid = paper["arxiv_id"]
        if pid in self.wrong_scope:
            return False
        if pid in self.listed:
            return True
        if self.rule and rule_matches(self.rule, paper):
            return self.surveys_count or not is_survey(paper)
        return False


# --------------------------------------------------------------------------- metrics
def unique(ids: list[str]) -> list[str]:
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def paper_index(trace: dict) -> dict[str, dict]:
    """Every paper seen anywhere in the trace, by arXiv ID, with the metadata rules need."""
    papers = {}
    for call in trace.get("qdrant_calls", []):
        for h in call["hits"]:
            papers.setdefault(h["arxiv_id"], h)
    return papers


def stage_ranks(trace: dict, pid: str) -> dict:
    """R6: where one relevant paper was at each stage, and where it was lost."""
    st = trace["stages"]
    pos = lambda lst: lst.index(pid) + 1 if pid in lst else None
    q_topic, q_survey = pos(st["qdrant"].get("topic", [])), pos(st["qdrant"].get("survey", []))
    rr = pos(st["rerank_papers"]) or pos(st["rerank_surveys"])
    pool = "paper" if pid in st["rerank_papers"] else ("survey" if pid in st["rerank_surveys"] else None)
    ranks = {"qdrant_topic": q_topic, "qdrant_survey": q_survey, "pool": pool, "rerank": rr,
             "sent_to_llm": pos(st["sent_to_llm"]), "llm_rank": pos(st["llm_ranked"]), "shown": pos(st["shown"])}
    if ranks["shown"]:
        ranks["dropped_at"] = None
    elif not (q_topic or q_survey):
        ranks["dropped_at"] = "qdrant"
    elif trace.get("mode") == "agent":   # no survey pool and no LLM ranking: the tools return the top k
        ranks["dropped_at"] = ("rerank (not in the top k returned to the agent)" if not ranks["sent_to_llm"]
                               else "agent (returned to the model but not cited)")
    elif pool == "survey":
        ranks["dropped_at"] = "rerank (survey pool: below the top 5 shown)"
    elif not ranks["sent_to_llm"]:
        ranks["dropped_at"] = "rerank (not in the top N+1/N+2 sent to the LLM)"
    else:
        ranks["dropped_at"] = "llm_ranking (sent but not picked)"
    return ranks


def ndcg_at_10(trace: dict, judge: Judge, papers: dict) -> float | None:
    if not judge.enumerated:
        return None
    rows = [c for c in trace["candidates"] if c.get("rerank_score") is not None]
    order = unique([c["arxiv_id"] for c in sorted(rows, key=lambda c: -c["rerank_score"])])[:10]
    dcg = sum(1 / math.log2(i + 2) for i, pid in enumerate(order) if judge.relevant(papers.get(pid, {"arxiv_id": pid})))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(judge.listed), 10)))
    return round(dcg / ideal, 4) if ideal else None


def parse_check(case: dict, trace: dict) -> dict | None:
    """Q1, rule part: the filters. The topic needs a human (it's free text), so it goes to the audit sheet."""
    if not case["q1_scored"] or not trace.get("parse"):
        return None
    exp, got = case["expected_parse"], trace["parse"]
    got_from = got["year_from"] if got["year"] is None else got["year"]
    got_to = got["year_to"] if got["year"] is None else got["year"]
    exp_author = (exp["author"] or "").strip().lower() or None
    got_authors = [a.strip().lower() for a in got["authors"]]
    checks = {"year_from": got_from == exp["year_from"], "year_to": got_to == exp["year_to"],
              "author": (exp_author is None and not got_authors) or (exp_author is not None and got_authors == [exp_author])}
    return {"filters_correct": all(checks.values()), "checks": checks,
            "expected_topic": exp["topic"], "got_topic": got["topic"]}


def score_case(case: dict, trace: dict) -> dict:
    judge = Judge(case)
    row = {"case_id": case["case_id"], "split": case["split"], "slice": case["slice"], "source": case["source"],
           "expected_behavior": case["expected_behavior"], "evaluator": "rule:" + EVALUATOR_VERSION}
    if trace.get("crashed"):
        return dict(row, crashed=True, failure={"label": "system_error", "reason": trace["errors"][0]},
                    latency_s=trace.get("latency_s"))

    papers = paper_index(trace)
    for p in trace["response"]["shown"]:
        papers.setdefault(p["arxiv_id"], p)
    get = lambda pid: papers.get(pid, {"arxiv_id": pid})
    st = trace["stages"]
    shown = unique(st["shown"])
    shown_papers = unique(st["shown_papers"])
    candidates = unique(st["qdrant"].get("topic", []) + st["qdrant"].get("survey", []))
    rel_shown = [pid for pid in shown if judge.relevant(get(pid))]

    if judge.enumerated:
        n_rel = len(judge.listed)
        found = lambda lst: len([p for p in lst if p in judge.listed and p not in judge.wrong_scope])
        recall_shown = round(found(shown) / n_rel, 4)
        recall_papers = round(found(shown_papers) / n_rel, 4)
        cand_recall = round(found(candidates) / n_rel, 4)
        first = next((i for i, pid in enumerate(shown, start=1) if judge.relevant(get(pid))), None)
        rr = round(1 / first, 4) if first else 0.0
        ranks = {pid: stage_ranks(trace, pid) for pid in sorted(judge.listed)}
    else:
        recall_shown = recall_papers = cand_recall = rr = None
        ranks = {}
    precision = round(len(rel_shown) / len(shown), 4) if shown and (judge.listed or judge.rule) else None
    wrong_scope_shown = [pid for pid in shown if pid in judge.wrong_scope]

    answered = trace["gate"]["answered"]
    positive = case["sufficiency_label"]
    gate_cell = {(True, True): "TP", (False, True): "FN", (True, False): "FP", (False, False): "TN"}[(answered, positive)]
    q1 = parse_check(case, trace)

    # Earliest defect (§8, §52), from rule-checkable evidence only; answer/attribution need the human audit
    if case["expected_behavior"] == "abstain":
        failure = ("abstention_ok", "The system returned no papers.") if not answered else \
                  ("abstention_fail", f"Expected an abstention, but {len(shown)} papers were shown (no gate in search mode).")
    elif trace["errors"] and not shown:
        failure = ("parse" if q1 and not q1["filters_correct"] else "retrieval_or_selection",
                   f"No papers shown: {trace['errors'][0][:160]}")
    elif q1 and not q1["filters_correct"]:
        bad = [k for k, ok in q1["checks"].items() if not ok]
        failure = ("parse", f"Filter parse wrong ({', '.join(bad)}): got {trace['parse']}, expected {case['expected_parse']}.")
    elif judge.enumerated and recall_shown == 0:
        lost = "; ".join(f"{pid} dropped at {r['dropped_at']} (qdrant topic #{r['qdrant_topic']}, "
                         f"survey #{r['qdrant_survey']}, rerank #{r['rerank']})" for pid, r in ranks.items())
        failure = ("retrieval_or_selection", f"No relevant paper shown: {lost}.")
    elif not judge.enumerated and (judge.listed or judge.rule) and not rel_shown:
        failure = ("retrieval_or_selection", f"None of the {len(shown)} shown papers is relevant under the rule/list.")
    else:
        failure = ("pass_retrieval", "Relevant evidence was shown; answer and attribution still need the human audit.")

    return dict(row, crashed=False, answered=answered, n_shown=len(shown), relevant_shown=rel_shown,
                wrong_scope_shown=wrong_scope_shown,
                retrieval={"recall_shown": recall_shown, "recall_papers": recall_papers,
                           "candidate_recall_at_20": cand_recall, "precision_shown": precision, "rr": rr,
                           "ndcg_at_10": ndcg_at_10(trace, judge, papers), "stage_ranks": ranks},
                gate={"cell": gate_cell, "answered": answered, "sufficiency_label": positive},
                parse=q1, failure={"label": failure[0], "reason": failure[1]},
                latency_s=trace.get("latency_s"), ranking_fallback=trace["response"]["ranking_fallback"])


# --------------------------------------------------------------------------- aggregation
def mean_metric(rows: list[dict], key: str) -> dict:
    vals = [r["retrieval"][key] for r in rows if not r["crashed"] and r["retrieval"][key] is not None]
    return {"mean": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals),
            "excluded": len(rows) - len(vals)}


def aggregate(rows: list[dict]) -> dict:
    keys = ["recall_shown", "recall_papers", "candidate_recall_at_20", "precision_shown", "rr", "ndcg_at_10"]
    ok = [r for r in rows if not r["crashed"]]
    q1 = [r["parse"]["filters_correct"] for r in ok if r.get("parse")]
    return {"n_cases": len(rows), "crashed": len(rows) - len(ok),
            "metrics": {k: mean_metric(rows, k) for k in keys},
            "q1_filters_correct": {"k": sum(q1), "n": len(q1)},
            "gate": dict(Counter(r["gate"]["cell"] for r in ok)),
            "failure_labels": dict(Counter(r["failure"]["label"] for r in rows))}


def hypothesis(rows: list[dict]) -> dict:
    """H1 (§5): Recall >= 60% on held-out ls_specific test cases. Case pass = every target shown (recall = 1)."""
    rs = [r for r in rows if r["split"] == "test" and r["slice"] == H1_SLICE and not r["crashed"]
          and r["retrieval"]["recall_shown"] is not None]
    k = sum(1 for r in rs if r["retrieval"]["recall_shown"] == 1.0)
    n = len(rs)
    ci = wilson(k, n)
    return {"H1": {"population": f"test split, slice {H1_SLICE}", "case_pass": "recall_shown == 1",
                   "k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": ci,
                   "mean_recall": round(sum(r["retrieval"]["recall_shown"] for r in rs) / n, 4) if n else None,
                   "supported": (ci[0] >= H_THRESHOLD) if ci else None,
                   "note": "supported only if the lower Wilson bound >= 60%"},
            "H2": {"status": "needs the human audit of citations (audit.csv); search mode has no cited summary"}}


def protocol_version() -> str:
    m = re.search(r"\*\*Status:\*\*\s*([^.(]+)", PROTOCOL.read_text(encoding="utf-8")) if PROTOCOL.exists() else None
    return m.group(1).strip() if m else "unknown"


AUDIT_FIELDS = ["case_id", "split", "slice", "expected_behavior", "message_sent", "parsed_topic", "expected_topic",
                "filters_correct", "shown", "relevant_shown", "failure_label", "failure_reason",
                "why_best", "criteria",
                # filled by the auditor (yes / no / n.a. + note)
                "topic_parse_ok", "A1_correct", "A2_grounded", "A3_relevant", "A4_complete", "A5_abstention_ok",
                "claims_cited_ok", "extra_relevant_ids", "human_agrees_with_label", "auditor", "note"]


def audit_row(case: dict, trace: dict, row: dict) -> dict:
    shown = trace.get("response", {}).get("shown", []) if not trace.get("crashed") else []
    return {"case_id": case["case_id"], "split": case["split"], "slice": case["slice"],
            "expected_behavior": case["expected_behavior"], "message_sent": trace.get("message_sent"),
            "parsed_topic": (trace.get("parse") or {}).get("topic"), "expected_topic": case["expected_parse"]["topic"],
            "filters_correct": (row.get("parse") or {}).get("filters_correct"),
            "shown": " | ".join(f"{p['arxiv_id']} ({p['year']}) {p['title']}" for p in shown),
            "relevant_shown": "; ".join(row.get("relevant_shown", [])),
            "failure_label": row["failure"]["label"], "failure_reason": row["failure"]["reason"],
            "why_best": trace.get("response", {}).get("why_best"),
            "criteria": "; ".join(trace.get("response", {}).get("criteria", []))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    traces = [json.loads(l) for l in (run / "traces.jsonl").open(encoding="utf-8") if l.strip()]
    cases = {c["case_id"]: c for f in sorted(CASES.glob("*-v*.jsonl"))
             for c in (json.loads(l) for l in f.open(encoding="utf-8") if l.strip())}
    missing = [t["case_id"] for t in traces if t["case_id"] not in cases]
    if missing:
        sys.exit(f"Traces for unknown cases {missing}: the case files changed since the run.")
    changed = [t["case_id"] for t in traces if t["case_set_version"] != cases[t["case_id"]]["case_set_version"]]
    if changed:
        sys.exit(f"Case version changed since the run for {changed}: rerun the system (§26).")

    rows = [score_case(cases[t["case_id"]], t) for t in traces]
    with (run / "per_case.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_slice = defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)
    heat = defaultdict(Counter)
    for r in rows:
        heat[r["slice"]][r["failure"]["label"]] += 1
    system = json.loads((run / "system_version.json").read_text(encoding="utf-8"))
    report = {
        "run_id": run.name,
        "system_version": {k: system.get(k) for k in ("git_commit", "input_variant", "qdrant", "reranker_model", "llm")},
        "evaluator_version": {"scorer": EVALUATOR_VERSION, "protocol": protocol_version(),
                              "cutoffs": {"shown": "<=3 papers + <=5 surveys", "candidates": "20 per Qdrant search",
                                          "ndcg": 10},
                              "aggregation": "macro mean over cases; None values excluded and counted",
                              "judge": None, "human_auditors": "see audit.csv"},
        "overall": aggregate(rows),
        "by_split": {s: aggregate([r for r in rows if r["split"] == s]) for s in sorted({r["split"] for r in rows})},
        "by_slice": {s: aggregate(rs) for s, rs in sorted(by_slice.items())},
        "slice_x_failure": {s: dict(c) for s, c in sorted(heat.items())},
        "hypotheses": hypothesis(rows),
        "latency_s": {"mean": round(sum(r["latency_s"] or 0 for r in rows) / len(rows), 2) if rows else None},
        "notes": ["Answer (A1-A5) and attribution (C1-C4) metrics need the human audit in audit.csv.",
                  "No gate score exists in search mode: G2 (threshold) is not applicable; G1 uses answered = papers shown.",
                  "Dependence: LitSearch questions come from a few ACL/ICLR paper sets; slices share topics (§33)."],
    }
    (run / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    audit = run / "audit.csv"
    if audit.exists():
        print(f"{audit.name} exists (it may hold human labels): not overwritten; wrote audit_new.csv instead.")
        audit = run / "audit_new.csv"
    with audit.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        tr = {t["case_id"]: t for t in traces}
        for r in rows:
            w.writerow(audit_row(cases[r["case_id"]], tr[r["case_id"]], r))

    o = report["overall"]
    print(f"Run {run.name}: {o['n_cases']} cases, {o['crashed']} crashed")
    for k, v in o["metrics"].items():
        print(f"  {k:<24} {v['mean'] if v['mean'] is not None else '-':>7}  (n={v['n']}, excluded {v['excluded']})")
    print(f"  Q1 filters correct      {o['q1_filters_correct']['k']}/{o['q1_filters_correct']['n']}")
    print(f"  gate                    {o['gate']}")
    print(f"  failure labels          {o['failure_labels']}")
    h1 = report["hypotheses"]["H1"]
    if h1["n"]:
        print(f"  H1 ({h1['population']}): {h1['k']}/{h1['n']} = {h1['rate']:.0%}, Wilson 95% {h1['wilson95']}")
    print(f"Wrote per_case.jsonl, report.json, {audit.name} in {run}")


if __name__ == "__main__":
    main()
