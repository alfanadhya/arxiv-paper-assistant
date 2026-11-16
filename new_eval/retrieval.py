import csv, re, sys, time
from pathlib import Path

# To let python read the app files (prompt_cleaning.py, threeofus_main.py) in the folder above
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompt_cleaning import collapse_repeats, remove_symbols, parse_message, length_problem   # light: works on the laptop too

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.csv"
RESULTS_FILE = HERE / "results.csv"
K = 20   # papers fetched from Qdrant per search (K_RETRIEVE in the app)

MODE = "lab"  # "lab" or "fake"

# Ranking stages recorded for every case, in pipeline order
STAGES = ["raw", "cleaned", "paper_pool", "reranked", "sent_to_llm", "final"]


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)          # baris pertama dipakai sebagai nama kolom
        for row in reader:
            ids = row["relevant_id"].split(";")             # "a;b;c" -> ["a", "b", "c"]
            ids = [i.strip() for i in ids if i.strip()]     # buang spasi dan isian kosong
            cases.append({"case": len(cases) + 1,                      # row number in cases.csv
                          "split": (row.get("split") or "").strip(),   # dev / test
                          "goal": (row.get("goal") or "").strip(),     # what this case checks
                          "query": row["query"].strip(), "relevant_ids": ids})
    return cases


def search_fake(query, k=20):
    return ["1906.00300", "2002.08909", "2004.04906", "2007.00808"][:k]


# Change Qdrant results to a list of arxiv ids
def to_ids(hits):
    return [re.sub(r"v\d+$", "", h.arxiv_id) for h in hits]


# {arxiv_id: score} from a ranked list; a paper listed twice keeps its first (best) score
def first_score(ids, scores):
    result = {}
    for pid, score in zip(ids, scores):
        result.setdefault(pid, round(float(score), 4))
    return result


# Step A1 + A2 (same as run_turn in the app): clean symbols, then read the year= / author= keys.
# A message without search= / title= is searched as a topic (the app's AGENT=0 path); one that already has
# a key is parsed exactly as typed, so its key errors are the app's own.
def clean_and_parse(query):
    symbol_cleaned = remove_symbols(collapse_repeats(query))   # "Hiii!!!! 😩" -> "Hi!"
    parsed = parse_message(symbol_cleaned, None)
    if parsed.mode_key not in ("search", "title"):
        parsed = parse_message("search= " + symbol_cleaned, None)   # text + authors + years
    return symbol_cleaned, parsed


# The app's input checks before any search (run_turn): length, key errors, empty topic. Error text or "".
def input_error(query, parsed):
    problem = length_problem(query)
    if problem:
        return problem[0]
    if parsed.errors:
        return " ".join(parsed.errors)
    if not re.search(r"[^\W_]", parsed.text):
        return "search= has no words left after removing symbols."
    return ""


# Empty result: every field the pipelines return, so run_all can always read them
def empty_result():
    result = {"symbol_cleaned": "", "parsed_text": "", "author_stripped": "", "cleaned_query": "",
              "survey_query": "", "filters": "none", "errors": "", "dense_ranks": {}, "sparse_ranks": {},
              "rrf_scores": {}, "rerank_scores": {}, "surveys_shown": [], "picked_by": ""}
    for stage in STAGES:
        result[stage] = []
    return result


# Fake steps (laptop): symbol cleaning and key parsing are real, everything after is pretend
def pipeline_fake(query):
    r = empty_result()
    r["symbol_cleaned"], parsed = clean_and_parse(query)
    r["parsed_text"] = r["author_stripped"] = parsed.text
    r["errors"] = input_error(query, parsed)
    if r["errors"]:                        # rejected: nothing is searched
        return r
    r["cleaned_query"] = parsed.text.lower()                     # pura-pura LLM merapikan query
    r["survey_query"] = "survey of " + r["cleaned_query"]
    r["raw"] = search_fake(query)
    r["cleaned"] = search_fake(r["cleaned_query"])
    r["dense_ranks"] = {pid: i for i, pid in enumerate(r["cleaned"], start=1)}   # pura-pura
    r["sparse_ranks"] = {r["cleaned"][0]: 1}                                     # pura-pura
    r["paper_pool"] = r["cleaned"]
    r["reranked"] = list(reversed(r["paper_pool"]))              # pura-pura reranker membalik urutan
    r["sent_to_llm"] = r["reranked"][:5]                         # N+2
    r["final"] = r["sent_to_llm"][:3]                            # pura-pura LLM pilih 3 teratas
    r["picked_by"] = "fake"
    return r


# Real steps (lab): the same functions and order as run_query in threeofus_main.py
def pipeline_lab(retriever, query):
    from retriever_hybrid_rrf import MetadataIntent
    r = empty_result()

    # A1 + A2: clean symbols, read the keys
    r["symbol_cleaned"], parsed = clean_and_parse(query)
    r["parsed_text"] = parsed.text

    # A0: the app's input checks; a rejected message is not searched at all (not even the baseline)
    r["errors"] = input_error(query, parsed)
    if r["errors"]:
        return r

    # Baseline: the query exactly as typed, no cleaning, no filters
    r["raw"] = to_ids(retriever.retrieve(query, K, MetadataIntent()))

    # A4: author names removed from the topic (recorded only; analyze_query does it again inside)
    r["author_stripped"] = parsed.text
    if parsed.authors:
        r["author_stripped"], _ = app.strip_author_names(parsed.text, parsed.authors)

    # A5: LLM spell check / topic extraction
    check, rest = app.analyze_query(parsed.text, meta=parsed)
    a = check.analysis
    r["cleaned_query"] = a.corrected_query
    if check.errors:                       # the app stops here and searches nothing
        r["errors"] = " ".join(check.errors)
        return r

    # B2: filters (only from year= / author= keys)
    filters = app.build_filters(a)
    r["filters"] = app.describe_filters(a)

    # A6 + B3: two Qdrant searches (topic and "survey of ..."), each = dense + sparse fused by RRF
    survey_focus = bool(app.SURVEY_WORD_RE.search(rest))
    r["survey_query"] = f"survey of {app.topic_without_survey(a.corrected_query)}"
    main_hits = retriever.retrieve(a.corrected_query, K, filters)
    survey_hits = retriever.retrieve(r["survey_query"], K, filters)
    r["cleaned"] = to_ids(main_hits)
    r["dense_ranks"] = {pid: h.dense_rank for pid, h in zip(r["cleaned"], main_hits)}
    r["sparse_ranks"] = {pid: h.sparse_rank for pid, h in zip(r["cleaned"], main_hits)}
    r["rrf_scores"] = first_score(r["cleaned"], [h.vector_score for h in main_hits])   # Qdrant hybrid (RRF) score

    # Pool split: research papers vs surveys (title contains "survey")
    paper_pool = [] if survey_focus else [h for h in main_hits if not app.is_survey(h)]
    survey_pool, seen = [], set()
    for h in main_hits + survey_hits:
        if app.is_survey(h) and h.point_id not in seen:
            seen.add(h.point_id)
            survey_pool.append(h)
    r["paper_pool"] = to_ids(paper_pool)

    # B4: cross-encoder rerank, each pool separately
    papers_ranked, paper_scores = app.rerank_hits(a.corrected_query, paper_pool)
    surveys_ranked, survey_scores = app.rerank_hits(a.corrected_query, survey_pool)
    r["reranked"] = to_ids(papers_ranked)
    r["rerank_scores"] = first_score(r["reranked"], paper_scores)   # cross-encoder score (higher = better)

    # Relevance gate (same rule as run_query): best score of both pools below the threshold -> nothing shown
    best = app.best_rerank_score(paper_scores, survey_scores)
    if best is not None and best < app.RERANK_MIN_SCORE:
        r["errors"] = f"relevance gate: best rerank score {best:.4f} < {app.RERANK_MIN_SCORE}"
        r["picked_by"] = "gate"
        return r
    r["surveys_shown"] = to_ids(surveys_ranked[:app.SURVEYS_SHOWN])

    # B5: N+2 (or N+1 if the LLM context is too small)
    sent, _ = app.fit_candidates(papers_ranked, app.MAX_PAPERS, a.corrected_query)
    r["sent_to_llm"] = to_ids(sent)

    # B6: the LLM picks N papers = the final output
    response = app.SearchResponse(prompt=query, search_prompt=parsed.text)
    response.request = parsed.with_meta(parsed.text)
    response.filters = r["filters"]
    outcome = app.rank_pool(response, "paper", sent, a.corrected_query, app.MAX_PAPERS,
                            lambda msg: None, len(paper_pool))
    r["final"] = to_ids([sent[i - 1] for i in outcome.picks])
    r["picked_by"] = outcome.picked_by      # "llm", or "fallback" (LLM failed -> reranker top 3)
    return r


# Is relevant paper available? (0.0 sampai 1.0)
def recall_at_k(found, relevant_ids):
    if not relevant_ids:
        return None
    hit = 0
    for paper_id in relevant_ids:
        if paper_id in found:
            hit += 1
    return hit / len(relevant_ids)


# At which position does the first relevant paper appear?
def first_rank(found, relevant_ids):
    for position, paper_id in enumerate(found, start=1):   # position = 1, 2, 3, ...
        if paper_id in relevant_ids:
            return position
    return None


# Best rank of a relevant paper in a {arxiv_id: rank} dict (dense or sparse list); None if not there
def best_rank_in(ranks, relevant_ids):
    found = [ranks[pid] for pid in relevant_ids if ranks.get(pid) is not None]
    return min(found) if found else None


# Best score of a relevant paper in a {arxiv_id: score} dict, and the score of the #1 paper in that list
# (to compare against); None if not there
def target_and_top_score(scores, relevant_ids):
    found = [scores[pid] for pid in relevant_ids if pid in scores]
    target = max(found) if found else None
    top = max(scores.values()) if scores else None
    return target, top


# Run all cases, calculate metrics, and save to CSV. retriever=None means fake mode.
def run_all(cases, retriever=None):
    rows = []
    for c in cases:
        i = c["case"]
        start = time.perf_counter()           # total time for this case: input checks -> final LLM pick
        if MODE == "lab":
            result = pipeline_lab(retriever, c["query"])
        else:
            result = pipeline_fake(c["query"])
        seconds = round(time.perf_counter() - start, 2)

        rel = c["relevant_ids"]
        # Query at every step that changes it
        row = {"split": c["split"], "case": i, "goal": c["goal"], "query": c["query"], "relevant_ids": ";".join(rel),
               "symbol_cleaned": result["symbol_cleaned"], "parsed_text": result["parsed_text"],
               "author_stripped": result["author_stripped"], "cleaned_query": result["cleaned_query"],
               "survey_query": result["survey_query"], "filters": result["filters"], "errors": result["errors"]}

        # Paper lists at every step that changes them
        for stage in STAGES:
            found = result[stage]
            row[f"{stage}_rank"] = first_rank(found, rel)
            for k in [3, 5]:                                  # top 3 (= N) dan top 5 (= N+2)
                row[f"{stage}_recall@{k}"] = recall_at_k(found[:k], rel)
                row[f"{stage}_top{k}"] = ";".join(found[:k])

        # Inside the hybrid search: who found the relevant paper, dense (meaning) or sparse (keywords)?
        row["dense_rank"] = best_rank_in(result["dense_ranks"], rel)
        row["sparse_rank"] = best_rank_in(result["sparse_ranks"], rel)
        # Scores of the relevant paper, next to the score of the #1 paper in the same list
        row["rrf_score"], row["rrf_top1_score"] = target_and_top_score(result["rrf_scores"], rel)
        row["rerank_score"], row["rerank_top1_score"] = target_and_top_score(result["rerank_scores"], rel)
        row["n_sent_to_llm"] = len(result["sent_to_llm"])
        row["picked_by"] = result["picked_by"]
        row["surveys_shown"] = ";".join(result["surveys_shown"])
        row["time_s"] = seconds
        rows.append(row)
        print(f"case {i}: raw={row['raw_rank']} cleaned={row['cleaned_rank']} "
              f"reranked={row['reranked_rank']} final={row['final_rank']} ({row['picked_by']}) {seconds:.2f}s")

    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    cases = load_cases(CASES_FILE)
    # Optional: only one split, e.g.  python new_eval/retrieval.py --split dev
    if "--split" in sys.argv:
        split = sys.argv[sys.argv.index("--split") + 1]
        cases = [c for c in cases if c["split"] == split]
        print(f"Split '{split}': {len(cases)} cases")

    retriever = None
    if MODE == "lab":
        import threeofus_main as app
        retriever = app.get_retriever()
        # Load the reranker and the Ollama model before timing (the app does this at startup too),
        # so case 1 is not charged for model loading
        for line in app.health_check():
            print("[warm-up]", line)

    rows = run_all(cases, retriever)
    print("Done. Results saved to", RESULTS_FILE)
