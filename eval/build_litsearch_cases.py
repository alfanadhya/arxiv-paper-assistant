"""Build LitSearch evaluation cases for the arXiv Paper Assistant (EVALUATION_PROTOCOL.md §6.4).

Steps (each step's counts go into the inclusion report, Lesson 6 §7):
  1. Load the 597 LitSearch questions (Hugging Face) and map each target paper's
     Semantic Scholar corpus ID to an arXiv ID. Both are cached in eval/data/.
  2. Look up every target in OUR corpus (the 1.1M-paper sub/*.jsonl files that were indexed).
     A target must exist there with a title and abstract, or the question is dropped.
  3. Check the mapping: the arXiv title must match the Semantic Scholar title.
  4. At most one question per target paper (dependence, Lesson 6 §33).
  5. Stratified random sample with a fixed seed; then a stratified dev/test split.

Run from the arxiv-paper-assistant folder:
    python -X utf8 eval/build_litsearch_cases.py --sub-dir "C:/Users/COM/Documents/RAG course-20261109T004859Z-1-001/sub"
"""

import argparse
import difflib
import glob
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

EVAL = Path(__file__).resolve().parent
DATA = EVAL / "data"
CASES = EVAL / "cases"
UA = {"User-Agent": "litsearch-eval-builder/0.1", "Content-Type": "application/json"}

CASE_SET_VERSION = "litsearch-v1"
CORPUS_VERSION = "sebun_bge768_hybrid_full (1,104,538 points; sub/*.jsonl)"
SEED = 2026
N_SPECIFIC, N_BROAD = 32, 8          # 40 cases; H1 is computed on the specific slice
DEV_SHARE = 0.30                     # about 12 dev / 28 test
TITLE_MATCH_MIN = 0.90               # similarity needed to accept the arXiv mapping


def get_json(url, data=None):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"rate limited: {url}")


def load_queries():
    cache = DATA / "litsearch_queries.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = []
    for off in range(0, 597, 100):
        d = get_json("https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/LitSearch"
                     f"&config=query&split=full&offset={off}&length=100")
        rows += [r["row"] for r in d["rows"]]
    for i, r in enumerate(rows):
        r["litsearch_row"] = i
    cache.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def load_s2_map(corpus_ids):
    cache = DATA / "litsearch_s2_map.json"
    known = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    missing = [c for c in corpus_ids if str(c) not in known]
    for i in range(0, len(missing), 500):
        batch = missing[i:i + 500]
        try:
            res = get_json("https://api.semanticscholar.org/graph/v1/paper/batch?fields=externalIds,title,year",
                           data=json.dumps({"ids": [f"CorpusId:{c}" for c in batch]}).encode())
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            # Semantic Scholar rejects IDs it cannot resolve: record them so they are dropped in step 2a.
            res = [None] * len(batch)
        for c, p in zip(batch, res):
            known[str(c)] = {"arxiv": ((p or {}).get("externalIds") or {}).get("ArXiv"),
                             "title": (p or {}).get("title"), "year": (p or {}).get("year")}
        time.sleep(3)
    cache.write_text(json.dumps(known, ensure_ascii=False, indent=1), encoding="utf-8")
    return known


def scan_corpus(sub_dir, wanted):
    """Find the wanted arXiv IDs in the indexed data files; keep only what a case needs."""
    found = {}
    for f in sorted(glob.glob(str(Path(sub_dir) / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in wanted:
                found[r["id"]] = {"title": " ".join((r.get("title") or "").split()),
                                  "has_abstract": bool((r.get("abstract") or "").strip()),
                                  "categories": r.get("categories", ""),
                                  "file": Path(f).name}
    return found


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", required=True, help="folder with the indexed sub/*.jsonl files")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    report = {"case_set_version": CASE_SET_VERSION, "corpus_version": CORPUS_VERSION, "seed": SEED,
              "built_on": str(date.today()), "steps": []}

    def step(name, rows):
        report["steps"].append({"step": name, "questions": len(rows)})
        print(f"{name:<55} {len(rows):>4} questions")

    queries = load_queries()
    step("1. LitSearch questions", queries)

    s2 = load_s2_map(sorted({c for q in queries for c in q["corpusids"]}))
    q = [x for x in queries if all(s2.get(str(c), {}).get("arxiv") for c in x["corpusids"])]
    step("2a. all targets have an arXiv ID", q)

    wanted = {s2[str(c)]["arxiv"] for x in q for c in x["corpusids"]}
    corpus = scan_corpus(args.sub_dir, wanted)
    q = [x for x in q if all(s2[str(c)]["arxiv"] in corpus and corpus[s2[str(c)]["arxiv"]]["has_abstract"]
                             for c in x["corpusids"])]
    step("2b. all targets in our corpus, with an abstract", q)

    def match(c):
        a = s2[str(c)]["arxiv"]
        return difflib.SequenceMatcher(None, norm(s2[str(c)]["title"]), norm(corpus[a]["title"])).ratio()
    q = [x for x in q if all(match(c) >= TITLE_MATCH_MIN for c in x["corpusids"])]
    step(f"3. arXiv title matches S2 title (>= {TITLE_MATCH_MIN})", q)

    rng = random.Random(SEED)
    rng.shuffle(q)
    used, dedup = set(), []
    for x in q:
        t = {s2[str(c)]["arxiv"] for c in x["corpusids"]}
        if t & used:
            continue
        used |= t
        dedup.append(x)
    step("4. at most one question per target paper", dedup)

    # 5. stratified sample: within each slice, spread across query_set as evenly as possible
    def sample(pool, n):
        by_set = defaultdict(list)
        for x in pool:
            by_set[x["query_set"]].append(x)
        picked = []
        while len(picked) < n and any(by_set.values()):
            for s in sorted(by_set):
                if by_set[s] and len(picked) < n:
                    picked.append(by_set[s].pop())
        return picked
    chosen = sample([x for x in dedup if x["specificity"] == 1], N_SPECIFIC) + \
             sample([x for x in dedup if x["specificity"] == 0], N_BROAD)
    step("5. stratified sample", chosen)

    cases = []
    for i, x in enumerate(chosen, 1):
        ids = [s2[str(c)]["arxiv"] for c in x["corpusids"]]
        specific = x["specificity"] == 1
        # Same schema as every other case file: see case_schema.py
        cases.append({
            "case_id": f"ls{i:03d}",
            "case_set_version": CASE_SET_VERSION,
            "corpus_version": CORPUS_VERSION,
            "split": None,
            "source": f"litsearch:{x['query_set']}",
            "coverage_purpose": "estimate",
            "slice": "ls_specific" if specific else "ls_broad",
            "hard_case_tags": ["multi_source"] if len(ids) > 1 else [],
            "hypothesis": ("A representative literature-search request: the target paper appears in the top 5"
                           if specific else
                           "A representative broad literature-search request: labelled papers appear among the results"),
            "request": {"query": x["query"], "audience": "graduate student",
                        "date_scope": None, "allowed_sources": "indexed corpus only"},
            "reference": {
                "relevant_ids": ids,
                "relevance_rule": None,
                "acceptable_alternative_ids": [],
                "wrong_scope_ids": [],
                "required_facts": [f"Cites {a} ({corpus[a]['title']})" for a in ids],
            },
            "expected_behavior": "answer",
            "expected_parse": {"topic": None, "year_from": None, "year_to": None, "author": None},
            "q1_scored": False,
            "sufficiency_label": True,
            "surveys_count": False,
            "rubric_id": "relevance-binary-v1",
            "label_author": "LitSearch (Ajith et al., EMNLP 2024); arXiv mapping by script",
            "label_date": str(date.today()),
            "label_notes": (f"LitSearch row {x['litsearch_row']}, quality={x['quality']} "
                            f"({'good' if x['quality'] == 2 else 'acceptable'}), specificity={x['specificity']} "
                            f"({'specific: <=5 papers fit' if specific else 'broad: <=20 papers fit; LitSearch labels only some of them'}). "
                            f"S2 corpus IDs {x['corpusids']}. Topic parse not labelled (Q1 not scored); "
                            "expected year/author filters are empty, so any invented filter is a parse error."),
            "source_details": {"litsearch_row": x["litsearch_row"], "quality": x["quality"],
                               "specificity": x["specificity"], "corpusids": x["corpusids"]},
        })

    # stratified dev/test split by slice (Lesson 6 §26), assigned before any system run
    for sl in ("ls_specific", "ls_broad"):
        group = [c for c in cases if c["slice"] == sl]
        rng.shuffle(group)
        n_dev = round(len(group) * DEV_SHARE)
        for j, c in enumerate(group):
            c["split"] = "dev" if j < n_dev else "test"

    out = CASES / f"{CASE_SET_VERSION}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report["final"] = {
        "cases": len(cases),
        "by_slice_split": dict(Counter(f"{c['slice']}/{c['split']}" for c in cases)),
        "by_query_set": dict(Counter(c["source"] for c in cases)),
        "by_quality": dict(Counter(c["source_details"]["quality"] for c in cases)),
        "multi_target_cases": sum(len(c["reference"]["relevant_ids"]) > 1 for c in cases),
        "target_files": dict(Counter(corpus[a]["file"] for c in cases for a in c["reference"]["relevant_ids"])),
    }
    (CASES / f"{CASE_SET_VERSION}_inclusion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n", json.dumps(report["final"], indent=1))

    print("\n--- spot check: 10 random cases (is the arXiv paper really what the question asks for?) ---")
    for c in random.Random(SEED + 1).sample(cases, 10):
        print(f"\n{c['case_id']} [{c['slice']}/{c['split']}] Q: {c['request']['query']}")
        for fact in c["reference"]["required_facts"]:
            print("   ->", fact)


if __name__ == "__main__":
    main()
