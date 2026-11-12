# Running the evaluation

This must run on the PC that runs the app: it needs Qdrant with `sebun_bge768_hybrid_full`, Ollama, and PyTorch. Scoring works on any PC.

## 1. Before you start

- Pull the latest `main`.
- Use the same `.venv` and `.env` as the app. The `.env` must have `OLLAMA_MODEL` set.
- Qdrant and Ollama must be running.
- Commit any local code changes first. Every run records the git commit, and it warns you about uncommitted changes.

## 2. Smoke test (2 cases, about a minute)

```powershell
.venv\Scripts\python.exe -X utf8 eval/run_eval.py --split dev --limit 2
```

You should see `[1/2] ... N shown` and `[2/2] ... N shown`. The first run downloads the embedding model and the reranker.

## 3. The dev run (22 cases)

```powershell
.venv\Scripts\python.exe -X utf8 eval/run_eval.py --split dev
```

If it stops, run the same command again with `--run-id <folder name>`. It skips the cases already done.

**Don't run `--split test`.** The test split is run once, on the frozen configuration, and only after the team agrees (protocol §9).

## 4. Send back the run folder

Each run creates a new folder, `eval/runs/<run_id>/`. Commit it, or zip it and send it. It holds:

| File | Contents |
|---|---|
| `system_version.json` | Git commit, models, collection point count, settings |
| `traces.jsonl` | Everything the system did for each case |
| `run_log.txt` | Console output |

## 5. Scoring (any PC, no GPU)

```powershell
python -X utf8 eval/score.py eval/runs/<run_id>
```

This writes `per_case.jsonl`, `report.json` and `audit.csv` into the run folder. `audit.csv` is for the human review. Rerunning the scorer never overwrites a filled-in `audit.csv`; it writes `audit_new.csv` instead.
