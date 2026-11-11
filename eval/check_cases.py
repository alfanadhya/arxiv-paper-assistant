"""Contract checks on all case files BEFORE any scoring (EVALUATION_PROTOCOL.md §10, Lesson 6 §55, §58).

Checks: every case follows the one schema (case_schema.py); case IDs are unique across files;
split and slice totals; and, with --sub-dir, that every referenced arXiv ID exists in the corpus.

Run from the arxiv-paper-assistant folder:
    python -X utf8 eval/check_cases.py
    python -X utf8 eval/check_cases.py --sub-dir "C:/Users/COM/Documents/RAG course-20261109T004859Z-1-001/sub"
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from case_schema import validate

CASES = Path(__file__).resolve().parent / "cases"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", help="corpus folder; if given, every referenced ID must exist there")
    args = ap.parse_args()

    files = sorted(CASES.glob("*-v*.jsonl"))
    cases = [(f.name, json.loads(l)) for f in files for l in f.open(encoding="utf-8") if l.strip()]
    problems = [e for _, c in cases for e in validate(c)]

    dup = [cid for cid, n in Counter(c["case_id"] for _, c in cases).items() if n > 1]
    if dup:
        problems.append(f"duplicate case_id: {dup}")
    keysets = {name: set().union(*(c.keys() for n, c in cases if n == name)) for name in {n for n, _ in cases}}
    if len({frozenset(k) for k in keysets.values()}) > 1:
        problems.append(f"case files have different field sets: {keysets}")

    if args.sub_dir:
        wanted = {i for _, c in cases for k in ("relevant_ids", "acceptable_alternative_ids", "wrong_scope_ids")
                  for i in c["reference"][k]}
        found = set()
        for f in glob.glob(str(Path(args.sub_dir) / "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r.get("id") in wanted and ((r.get("title") or "").strip() or (r.get("abstract") or "").strip()):
                    found.add(r["id"])
        missing = sorted(wanted - found)
        if missing:
            problems.append(f"{len(missing)} referenced IDs not in the corpus: {missing[:10]}")
        print(f"referenced IDs found in corpus: {len(found)}/{len(wanted)}")

    for name in sorted({n for n, _ in cases}):
        cs = [c for n, c in cases if n == name]
        print(f"{name}: {len(cs)} cases | split {dict(Counter(c['split'] for c in cs))} "
              f"| slices {len(set(c['slice'] for c in cs))}")
    print(f"total: {len(cases)} cases | split {dict(Counter(c['split'] for _, c in cases))}")
    if problems:
        raise SystemExit("CONTRACT CHECK FAILED:\n  " + "\n  ".join(problems))
    print("contract check passed: one schema, unique IDs" + (", all IDs in corpus" if args.sub_dir else ""))


if __name__ == "__main__":
    main()
