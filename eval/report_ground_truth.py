"""Show the result of a ground-truth run (eval/run_ground_truth.py) as one table row per paper:

    Paper | Title | Changed title | Qdrant rank | Reranked rank | Sent to LLM | Selected

Title is the text that was sent; Changed title is the query the app actually searched Qdrant with (several
searches are joined with ' ; '). The ranks are the 1-based position of the target paper at each stage, '-' if absent.

Run from the project folder:
    python -X utf8 eval/report_ground_truth.py                     # the newest ground-truth run
    python -X utf8 eval/report_ground_truth.py <run_id or folder>  # a given run

The same table is saved as report.tsv in the run folder (opens in Excel).
"""

import argparse
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
STAGES = [("qdrant", "Qdrant rank"), ("reranked", "Reranked rank"), ("sent_to_llm", "Sent to LLM"),
          ("llm_selected", "Selected")]
HEADER = ["Paper", "Title", "Changed title"] + [label for _, label in STAGES]
WIDTH = 50   # console width of the two title columns (report.tsv has them in full)


def find_run(name: str | None) -> Path:
    if name:
        run = Path(name) if Path(name).exists() else RUNS / name
    else:   # newest run folder that has ground-truth records
        runs = [d for d in RUNS.glob("*ground_truth*") if (d / "records.jsonl").exists()]
        if not runs:
            raise SystemExit(f"No ground-truth run found in {RUNS}")
        run = max(runs, key=lambda d: (d / "records.jsonl").stat().st_mtime)
    if not (run / "records.jsonl").exists():
        raise SystemExit(f"No records.jsonl in {run}")
    return run


def row_of(r: dict) -> list[str]:
    if r.get("crashed"):
        return [r["target_id"], r["message_sent"], "CRASHED: " + r["errors"][0]] + ["-"] * len(STAGES)
    rank = r["target_rank"]
    return [r["target_id"], r["message_sent"], " ; ".join(r["modified_queries"])] + \
           [str(rank[stage]) if rank[stage] else "-" for stage, _ in STAGES]


def cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 3] + "..."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run id or run folder (default: the newest ground-truth run)")
    args = ap.parse_args()

    run = find_run(args.run)
    records = [json.loads(l) for l in (run / "records.jsonl").open(encoding="utf-8") if l.strip()]
    rows = [row_of(r) for r in records]

    (run / "report.tsv").write_text("\n".join("\t".join(row) for row in [HEADER] + rows) + "\n", encoding="utf-8")

    shown = [[row[0], cut(row[1], WIDTH), cut(row[2], WIDTH)] + row[3:] for row in [HEADER] + rows]
    widths = [max(len(row[i]) for row in shown) for i in range(len(HEADER))]
    print(f"Run: {run.name}  ({len(records)} papers)\n")
    for n, row in enumerate(shown):
        print("  ".join(cell.ljust(w) if i < 3 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths))))
        if n == 0:
            print("  ".join("-" * w for w in widths))

    # Summary: in how many papers the target reached each stage, and how often it was the first one
    ok = [r for r in records if not r.get("crashed")]
    print(f"\nTarget paper reached the stage (of {len(records)} papers):")
    for stage, label in STAGES:
        ranks = [r["target_rank"][stage] for r in ok if r["target_rank"][stage]]
        print(f"  {label:<14} {len(ranks):>3}   at rank 1: {sum(1 for x in ranks if x == 1):>3}"
              + (f"   mean rank: {sum(ranks) / len(ranks):.1f}" if ranks else ""))
    changed = sum(1 for r in ok if " ; ".join(r["modified_queries"]).strip().lower() != r["message_sent"].strip().lower())
    print(f"  Title changed by the app before searching: {changed} of {len(ok)}")
    if len(ok) < len(records):
        print(f"  Crashed: {len(records) - len(ok)}")
    print(f"\nSaved: {run / 'report.tsv'}")


if __name__ == "__main__":
    main()
