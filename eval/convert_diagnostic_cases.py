"""Convert cases/diagnostic_cases_draft.csv (the editable spreadsheet) into cases/diagnostic-v1.jsonl,
in the same schema as the LitSearch cases (case_schema.py), and assign dev/test splits.

Splits (EVALUATION_PROTOCOL.md §6.5): stratified by slice with a fixed seed, so each slice gets
at least one dev case and at least one test case. Run this after the case review and before the first system run.
Rerunning it after the first run would reshuffle the splits, so don't.

Run from the arxiv-paper-assistant folder:
    python -X utf8 eval/convert_diagnostic_cases.py
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from case_schema import validate

EVAL = Path(__file__).resolve().parent
SRC = EVAL / "cases" / "diagnostic_cases_draft.csv"
CASE_SET_VERSION = "diagnostic-v1"
CORPUS_VERSION = "sebun_bge768_hybrid_full (1,104,538 points; sub/*.jsonl)"
SEED = 2026


def ids(cell):
    return [x.strip() for x in cell.split(";") if x.strip()]


def year(cell):
    return int(cell) if cell.strip() else None


def yes(cell):
    return cell.strip().lower() == "yes"


def to_case(r):
    date_scope = None
    if r["year_from"].strip() or r["year_to"].strip():
        date_scope = {"year_from": year(r["year_from"]), "year_to": year(r["year_to"])}
    return {
        "case_id": r["case_id"].strip(),
        "case_set_version": CASE_SET_VERSION,
        "corpus_version": CORPUS_VERSION,
        "split": None,
        "source": "manual",
        "coverage_purpose": "diagnostic",
        "slice": r["slice"].strip(),
        "hard_case_tags": ids(r["hard_case_tags"]),
        "hypothesis": r["hypothesis"].strip(),
        # query = the exact message typed (keys included); setup_turns = earlier messages, separated by " || "
        "request": {"query": r["query"].strip(),
                    "setup_turns": [t.strip() for t in r["setup_turns"].split("||") if t.strip()],
                    "audience": r["audience"].strip() or "graduate student",
                    "date_scope": date_scope, "allowed_sources": "indexed corpus only"},
        "reference": {
            "relevant_ids": ids(r["relevant_ids"]),
            "relevance_rule": json.loads(r["relevance_rule"]) if r["relevance_rule"].strip() else None,
            "acceptable_alternative_ids": ids(r["acceptable_alternative_ids"]),
            "wrong_scope_ids": ids(r["wrong_scope_ids"]),
            "required_facts": ids(r["required_facts"]),
        },
        "expected_behavior": r["expected_behavior"].strip(),
        "expected_parse": {"topic": r["expected_topic"].strip() or None,
                           "year_from": year(r["expected_year_from"]),
                           "year_to": year(r["expected_year_to"]),
                           "author": r["expected_author"].strip() or None},
        "q1_scored": True,
        "sufficiency_label": yes(r["sufficiency_label"]),
        "surveys_count": yes(r["surveys_count"]),
        "rubric_id": "relevance-binary-v1",
        "label_author": r["label_author"].strip(),
        "label_date": r["label_date"].strip(),
        "label_notes": r["label_notes"].strip(),
        "source_details": None,
    }


def main():
    cases = [to_case(r) for r in csv.DictReader(SRC.open(encoding="utf-8-sig"))]

    rng = random.Random(SEED)
    by_slice = defaultdict(list)
    for c in cases:
        by_slice[c["slice"]].append(c)
    for sl in sorted(by_slice):
        group = sorted(by_slice[sl], key=lambda c: c["case_id"])
        rng.shuffle(group)
        n_dev = max(1, len(group) // 2) if len(group) > 1 else 0
        for j, c in enumerate(group):
            c["split"] = "dev" if j < n_dev else "test"

    errors = [e for c in cases for e in validate(c)]
    if errors:
        raise SystemExit("schema errors:\n  " + "\n  ".join(errors))

    out = EVAL / "cases" / f"{CASE_SET_VERSION}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {out.relative_to(EVAL.parent)}")
    print("split by slice:", dict(sorted(Counter(f"{c['slice']}/{c['split']}" for c in cases).items())))
    print("totals:", dict(Counter(c["split"] for c in cases)))


if __name__ == "__main__":
    main()
