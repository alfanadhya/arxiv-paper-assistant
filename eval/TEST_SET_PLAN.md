# Test Set Plan: next additions (draft, 2026-11-13)

Working notes for extending the diagnostic test set. Most of this can be done on a laptop.
The official rules (slices, rubric, split) are in `EVALUATION_PROTOCOL.md` §6; this file only plans what to add.

## 0. Where things stand

| Item | State |
|---|---|
| Diagnostic cases | `cases/diagnostic-v1.jsonl`: 21 cases (m00–m20), 10 slices, one dev + one test per slice |
| LitSearch cases | `cases/litsearch-v1.jsonl`: 40 cases (12 dev / 28 test) |
| Runs | `runs/20261112-1053_dev_keyed` is the **only search-mode run** (14 cases). The `1339`/`1351` smoke runs and `1431` (22 cases) are **agent mode**: no N+2 → LLM ranking, `llm_ranked` is always empty |
| Context | Run `1053` used `num_ctx = 8192`; the code default is now 16384 (`threeofus_main.py:38`) |

## 1. Laptop vs PC

| Can do on the laptop (plain Python / text editor) | Needs the PC (Qdrant, models, Ollama, corpus) |
|---|---|
| Write new cases in `cases/*.jsonl` | New runs: `run_eval.py` |
| `python eval/check_cases.py` (schema only) | `check_cases.py --sub-dir <corpus>` (are the IDs in the corpus?) |
| `python eval/score.py` on existing runs | Anything that calls the search pipeline |
| Offline reranker-vs-LLM analysis of `traces.jsonl` | |
| Edit `EVALUATION_PROTOCOL.md`, `docs/` | |

Copy `eval/` (≈2.6 MB) and `docs/` to the laptop. When back on the PC, copy back **only the files you changed**.

## 2. New slices

Rule from the protocol: **one slice per case**, at least **one dev + one test** per slice, split assigned **before any run**.

### 2.1 `followup_rewrite`: does the follow-up rewrite retrieve the right paper?

The rewrite (`rewrite_with_history`, `threeofus_main.py:666`) runs only when there is history **and** the last message contains a follow-up cue (`FOLLOWUP_CUE_RE`, `:96`: it, that, those, the second one, instead, also, …). So every case is **multi-turn**, with turn 1 hardcoded, not generated live, so the case is repeatable.

| ID | Turn 1 (user) | Turn 2 (user, the scored query) | Expected |
|---|---|---|---|
| F1 | papers on BM25 / TF-IDF sparse retrieval for open-domain question answering | what about the dense version of that instead? | rewrite ≈ "dense retrieval for open-domain QA"; DPR `2004.04906` at #1 |
| F2 | retrieval for open-domain QA (assistant reply hardcoded as a list with DPR at [2]) | tell me more about the second one | DPR at #1 |
| F3 | open-domain question answering papers | which of those use a BERT dual-encoder instead of sparse vectors? | the topic from turn 1 is carried over; DPR in top 3 |
| F4 (control) | LoRA fine-tuning | is there also a dense passage retriever for open-domain QA that beats BM25? | "also" triggers the rewrite, but **no LoRA words** leak in; DPR at #1 |

**Measure** each case under 3 conditions and record DPR's rank at every stage (`score.py` `dropped_at`), plus the rewritten text:
- A: turn 2 as typed (no rewrite)
- B: the rewrite output
- C: oracle, a hand-written standalone query

The rewrite works if **B ≈ C and B > A**. If C also fails, the problem is retrieval, not the rewrite.
**Requirement:** turn 2 alone (condition A) must **not** find DPR. Otherwise the case tests retrieval, not rewriting.

**Code needed first (PC or laptop, but test on the PC):**
- `case_schema.py`: add `request.history` (list of `{role, content}` turns; `[]` for single-turn cases)
- `run_eval.py`: pass the history to the pipeline, and save the rewrite output in the trace
- add `followup_rewrite` to the slice table in `EVALUATION_PROTOCOL.md` §6.2

### 2.2 `seminal_vs_followup`: does the LLM pick the original paper over its follow-ups?

Made for the **reranker-vs-LLM** comparison (§3 below). Several close candidates must reach the top 5, and the request needs reasoning ("original … not later work"), which a cross-encoder scoring word overlap tends to miss.

| ID | Split | Query | relevant | wrong_scope |
|---|---|---|---|---|
| SV1 | dev | the original paper that trained a BERT dual-encoder on question-passage pairs and beat BM25 for open-domain question answering, not later work that improved its hard negatives | `2004.04906` DPR | `2007.00808` ANCE*, `2010.08191` RocketQA* |
| SV2 | test | the original paper that freezes the pretrained weights and trains small low-rank update matrices, not later work that quantizes the base model | `2106.09685` LoRA | `2305.14314` QLoRA |

Hypothesis (SV1): *Several dense-retrieval papers match the description; the LLM, unlike the cross-encoder, uses "original … not later work" to put DPR first and keep the follow-ups out.*

Rules: no author name and no `by`/`author`/`written` cue words (they would trigger the author filter and shrink the pool); `expected_parse.author = null`; `date_scope = null`.

### 2.3 `negation`: can the system handle "without X"?

| ID | Split | Query | relevant | wrong_scope |
|---|---|---|---|---|
| NG1 | dev | open-domain QA retrievers trained only on question-passage pairs, without extra pretraining like the inverse cloze task | `2004.04906` DPR; acceptable: ANCE*, RocketQA* | `1906.00300` ORQA*, `2002.08909` REALM* |
| NG2 | test | *(to write: another negation case on a different topic)* | | |

### 2.4 Extra cases for existing slices

| ID | Slice | Split | Query | Notes |
|---|---|---|---|---|
| A1 | `filter_author` | dev or test | what has Vladimir Karpukhin written on dense passage retrieval? | keyed mode adds `author=Vladimir Karpukhin`. Tests the author filter **and** that the name is removed from the topic (`strip_author_names`; the `m07` bug). DPR at #1; ANCE / RocketQA in `wrong_scope_ids`. **Not useful for the reranker-vs-LLM comparison**: the filtered pool is tiny, so the LLM has nothing to choose from |
| P1 | `direct_specific` | dev or test | using two BERT encoders to find Wikipedia passages for answering trivia questions, instead of TF-IDF | paraphrase without "dense"/"DPR"; DPR at #1 |
| K1 | `direct_specific` | (control) | dense passage retrieval for open-domain question answering | sanity check: should be a tie between reranker and LLM |

Balance dev and test when assigning splits (protocol §6.5).

`*` = arXiv ID written from memory, **not yet verified** against the corpus. On the PC, run `check_cases.py --sub-dir <corpus>` before freezing the labels. LoRA `2106.09685` and QLoRA `2305.14314` are already verified (used in m00).

## 3. Analysis: reranker top N vs LLM pick from N+2

No new run is needed for a first look. Each trace already has:
- `stages.rerank_papers`: reranker order. System **A** = `[:3]`
- `stages.sent_to_llm`: the 4–5 papers sent to the LLM
- `stages.shown_papers`: system **B**, the LLM's picks

Labels come from `reference` in the case files (match on `case_id`). Reuse `Judge.relevant` from `score.py` so "relevant" means the same thing everywhere.

Report:
1. Recall@3 and Precision@3 for A and B, with counts
2. Paired wins/losses/ties (B>A, B<A, =) and a sign test on the non-ties
3. **Rescue**: relevant papers at reranker rank 4–5 that the LLM picked. **Drop**: relevant papers at reranker rank 1–3 that the LLM left out
4. Cost: LLM latency and tokens

Only cases with **≥ 4 candidates** in the paper pool count (with ≤ 3, A = B automatically). Report per slice. Dev split only.
"LLM sees only N" is not a separate arm: picking 3 of 3 gives the same set as reranker top 3.

Reading DPR's rank in `rerank_papers` for SV1:
- 1: tie, the prompt is too easy
- 2–3: does the LLM move DPR up or drop it?
- 4–5: best case. DPR shows up only if the LLM rescues it
- \>5: the LLM never sees DPR. That's a reranker failure, and the case can't compare the two

Worked example (run `1053`, case `m01`): A = [Prefix-Tuning, Input-Tuning, Plug-Tagger]; B = [Prefix-Tuning, audio-captioning prefix (reranker #5), Input-Tuning]. The LLM dropped Plug-Tagger (sequence labeling, not generation) and rescued #5. Cost: 66.8 s, ranking output 6,791 tokens.

Possible script: `eval/compare_rerank_vs_llm.py` (reads `traces.jsonl` + case files, prints the tables above). It runs on the laptop.

## 4. Next run (on the PC)

- **Search mode** (`search=` prefix). Agent mode skips the LLM ranking
- All dev cases, including the new ones; 2–3 repeats, because the LLM is sampled (`temperature=0.6`)
- Record `num_ctx` in `system_version.json` (now 16384)
- Keep the test split untouched until the configuration is frozen

## 5. Open issues found while planning

- **Ranking output reserve is too small.** `RANKING_OUTPUT_RESERVE = 3000`, but `m01` produced 6,791 output tokens (99% of the 8,192 context, with the warning "input may be cut"). The N+2 fit check (`fit_pool`, `threeofus_main.py:1046`) therefore underestimates the real use.
- `followup_rewrite` cannot be evaluated until the schema and runner support `history` (§2.1).

## 6. Checklist

Laptop:
- [ ] Write the F1–F4, SV1–SV2, NG1–NG2, A1, P1, K1 cases as `.jsonl` (new case version, e.g. `diagnostic-v2`)
- [ ] Write NG2 (second negation case)
- [ ] Add the new slices to `EVALUATION_PROTOCOL.md` §6.2
- [ ] `check_cases.py` (schema)
- [ ] `compare_rerank_vs_llm.py` and try it on run `1053`

PC:
- [ ] `check_cases.py --sub-dir …` for the `*` IDs
- [ ] `request.history` in the schema and runner
- [ ] Search-mode dev run (2–3 repeats)
