# Developer notes

Notes on how `threeofus_main.py` and its helper files work, for whoever changes the code. For setup and how to use the app, see [README.md](README.md).

## Running

| How | Command |
|---|---|
| GUI | `python -m chainlit run threeofus_main.py -w -h --host 0.0.0.0 --port 8000`, then open http://localhost:8000 |
| CLI, one prompt | `python threeofus_main.py "search= your topic"` |
| CLI, with follow-ups | `python threeofus_main.py --chat` (an empty line quits) |

- `chainlit` alone may not be on PATH; `python -m chainlit` always works.
- The GUI is a web server, so the terminal stays busy until Ctrl+C.
- **`-w` (watch mode) only reloads `threeofus_main.py` itself.** Imported modules (`app.py`, `db.py`, `chatbot_diagnostics.py`, `prompt_cleaning.py`, `retriever_hybrid_rrf.py`) need a full restart.
- **Every reload or restart drops open browser connections.** The browser reconnects and the chat is resumed (see [Chat resume](#chat-resume)). In the log this shows as `ConnectionResetError: [WinError 10054]`, which is harmless. For a stable demo, run without `-w`.
- After changing `public/custom.js`, bump the `?v=` in `custom_js` in `.chainlit/config.toml` so browsers load the new file.

## Requirements

- **Ollama** running with `OLLAMA_MODEL` pulled.
- **Qdrant** at `QDRANT_URL` with the collection `COLLECTION` (`sebun_bge768_hybrid_full`).
- **`.env`:** `OLLAMA_MODEL`, `QDRANT_URL`, `CHAINLIT_AUTH_SECRET`. Keep it private; changing the secret logs everyone out. Optional settings are listed in the README.
- **Packages:** see `requirements.txt` (chainlit, SQLAlchemy + aiosqlite for chat history, langchain-ollama, qdrant-client, sentence-transformers, fastembed, langchain-qdrant, httpx for the zip export).

## Files

| File | Role |
|---|---|
| `threeofus_main.py` | Pipeline (`run_turn`, `run_query`, `run_title_query`, `run_analyze`), prompts, LLM chains, Chainlit handlers, `/signup` and `/prefs/newest-first` routes |
| `prompt_cleaning.py` | Input checks before any model call: `parse_message`, `length_problem`, `collapse_repeats`, `remove_symbols`, `ParsedMessage` |
| `chatbot_diagnostics.py` | `Diagnostics` buffer (tagged notes, rank trace, context usage), `format_diagnostics`, the token-usage callback |
| `retriever_hybrid_rrf.py` | Qdrant hybrid search (dense + BM25 with RRF) and `MetadataIntent` filters |
| `app.py` | GUI helpers: paper cards, Select buttons, reading-list panel, mode banner, session state, chat resume. Never imports `threeofus_main.py` |
| `db.py` | `history.db` schema, accounts, preferences, the per-user data layer (`CleanupDataLayer`), account commands |
| `export.py` | Everything Export writes: selected-chunks JSONL and the BibTeX + PDF zip |
| `public/custom.js` | Page tweaks: toasts, fresh chat after deleting all chats, reading-list tab, Newest first box, mode banner, green best match, sign-up link |
| `public/elements/ReadingList.jsx`, `Diagnostics.jsx` | The reading-list panel and the diagnostics panel |

## One message, step by step (`run_turn`)

**Input checks** (`prompt_cleaning.py`, no model call until all pass):
1. **Keys.**
   - **Mode keys:** `search=`, `title=`, `analyze=` (also `analyse=`). A mode key switches the chat's mode, which stays on until another mode key is typed. A message without a mode key uses the remembered mode, and its text before any metadata key is the value.
   - **Metadata keys:** `year=` (`2021`, `2019-2022`, `2019-`, `-2020`) and `author=` (`A; B`, or repeat the key). They apply to one message only, are the **only** source of years and authors, and are ignored by `analyze=`.
   - **Errors stop the turn:** several mode keys, a key used twice, an empty value, a bad year, an empty author.
2. **Length:** the message is at most `MAX_PROMPT_WORDS` (200). A `title=` value is `TITLE_MIN_WORDS`–`TITLE_MAX_WORDS` (3–30). Only words with a letter or digit count.
3. **Collapse:** repeated spaces, and runs of the same symbol (`...`, `???`, `;;`), become one.
4. **Symbols:** letters, digits and `, - : . ; ? '` are kept, other symbols become spaces. Text with no letter or digit left is rejected.

**Modes:**
- **New chats start in search mode.** `app.set_mode` saves the mode in the session and sends it to the banner.
- **A reopened chat keeps its mode.** Chats saved before modes existed get analyze if they have papers, else search.
- **No mode at all** (only possible for those old chats) shows the fixed syntax reminder (`command_reminder`). `CHAT_WITH_LLM = False`, so there's no free chat with the model; `chat_reply` is kept for later.

## Search (`run_query`)

1. **Follow-up rewrite:** only when there is history **and** the text matches `FOLLOWUP_CUE_RE`. One LLM call (`rewrite_with_history`, `[memory]` in diagnostics). The cue list is broad ("it", "more", "one", digits), so it also fires on normal topics like "one-shot learning".
2. **Spell check:** `analyze_query` → `run_analyzer`. One call with reasoning off, structured output `QueryAnalysis`. It retries once with a stronger prompt if the topic comes back empty, and falls back to `regex_fallback` if the model fails. **Years and authors guessed by the model are ignored.**
3. **`validate`:**
   - **Stops the search:** a year outside `DB_FIRST_YEAR`–`DB_LAST_YEAR` (coverage `DB_COVERAGE`), no topic left, the model marks the query unclear.
   - **Adjusts and continues:** a partly out-of-range year is trimmed, a reversed range is swapped.
4. **Survey focus:** the word "survey" in the text means only surveys are searched. Note that it's tested on the text *before* the spell fix.
5. **Retrieval** (`search_papers`): two Qdrant searches with `K_RETRIEVE` (20) each, one for the topic and one for "survey of <topic>". Year and author are **hard filters**.
   - Authors use `author_keys` (last name + first initial) when an initial is given, else `author_last_names`. If nothing matches with initials, it retries with last names only.
   - Several authors are combined as **AND**.
   - Hits are split by "survey" in the title.
6. **Reranking:** the `RERANKER_MODEL` cross-encoder (fp16 on GPU, cached in `models_cache/`) scores both pools.
7. **Papers:** the LLM ranks only the research papers (`rank_pool`, `RANK_POOL_PROMPT`, reasoning on). It sees N+2 of them (N = `MAX_PAPERS` = 3), or N+1 if the context is too small (`fit_pool`).
   - Invalid picks are cleaned out.
   - If the call fails, the papers keep the reranker order (`[ranking] FALLBACK (papers)`, picked_by `fallback`).
8. **Surveys:** no LLM call. The reranker's top `SURVEYS_SHOWN` (5) are shown, and the rank trace shows "–" in the LLM columns.
9. **Display order:** with the **Newest first** box ticked, papers and surveys are sorted by year. Without it, papers are in the LLM's order and surveys in the reranker's order.

## Title lookup (`run_title_query`)

- **Spell check:** the title prompt fixes spelling only and keeps the wording.
- **Candidates:** `K_TITLE` (100) come from the hybrid search. A `category:` with no results is dropped and the search runs again.
- **Rerank pool:** the `TITLE_POOL_BY_SIM` (10) most similar titles plus Qdrant's top `TITLE_POOL_BY_QDRANT` (20), scored by the reranker.
- **Scoring:** `title_similarity` (character ratio + word F1) is combined with the reranker score:
  - similarity ≥ `TITLE_EXACT` (0.95): exact match, always wins; the reranker only breaks ties
  - `TITLE_BLEND_MIN` (0.60) up to 0.95: 70% similarity + 30% reranker
  - below 0.60: 70% reranker + 30% similarity
- **Bonuses:** author and year only add small bonuses (`TITLE_AUTHOR_BONUS`, `TITLE_YEAR_BONUS`) and never filter. The year allows ±1, because the arXiv year often differs from the venue year.
- **Result:** one paper, with no LLM ranking. A non-exact match comes with a warning.

## Analyze (`run_analyze`)

- **What it can refer to:** after each successful search or title lookup, the handler stores the result's labels and arXiv ids in `last_results` (`app.set_last_results`).
- **Resolving references** (`resolve_paper_refs`, in code):
  - `1`, `[2]`, `S1` or `[S3]` are labels of the last result.
  - An arXiv id can be any paper in `papers_by_id`.
  - No reference means the whole last result, up to `ANALYZE_MAX_PAPERS` (8).
  - An unknown label stops with a list of the available ones.
  - A plain number always counts as a label: "top 3 ideas" means paper 3.
- **The LLM call:** one call (`ANALYZE_PROMPT`, reasoning on, capped at `ANALYZE_MAX_TOKENS`) with the papers' title, authors, year and abstract. It is told to answer only from those, cite labels, and not repeat titles.
- **Output:** the answer streams to the GUI. The papers used aren't printed; they go into the conversation memory (`memory_note`) and the `[analyze]` diagnostics line.

## LLM calls and LangChain

- **One Ollama model** for every call (`ollama_llm`), with the same `NUM_CTX` and `OLLAMA_KEEP_ALIVE`, so Ollama never reloads it between steps.
- **Templates and chains:** `ChatPromptTemplate`s chained with LCEL (`prompt | model | parser`), built once and cached.
- **Structured outputs:** `QueryAnalysis`, `PoolRanking` (`include_raw=True` keeps the thinking), `RewriteResult`. Free-text answers use `StrOutputParser` so they can stream.
- **Memory is our own:** the last `MAX_HISTORY_TURNS` (10) pairs in `cl.user_session["history"]`, passed in through `MessagesPlaceholder("history")`. `assistant_turn_text` decides what a turn stores:
  - search: labels, ids and titles
  - analyze: papers used + the answer, truncated to 2000 characters
- **Token usage:** the `ContextUsageHandler` callback records every call for the `[context]` lines.

## Diagnostics

- **Tagged lines**, in order: `[mode]`, `[query]` (parser and cleaning steps), `[analysis]`, `[validation]`, `[memory]`, `[rerank]`, `[ranking]`, `[analyze]`, `[context]`.
- **Rank trace:** one row per candidate, with Qdrant position, reranker rank and score, the label sent to the LLM, the LLM pick, and the displayed position.
- **Saved with the chat:** the panel data is stored in the diagnostics message's metadata. A plain-text digest is also printed to the console.

## Chat resume

- **What's saved:** Chainlit saves messages and the session (JSON values only, at most 1 MB per chat; above that the whole session is dropped).
- **What isn't saved:** buttons, and elements such as panels, because there's no storage provider.
- **`app.resume` rebuilds them:**
  - Select and Reading list buttons, from each message's metadata, with fixed ids so they never show twice.
  - Diagnostics panels, from each diagnostics message's metadata. These are added to `thread["elements"]`, not sent separately: Chainlit sends `thread` to the browser **after** `on_chat_resume`, and the browser then replaces all its elements with that list.
  - The reading list and the mode banner.
- **Paper limit:** `papers_by_id` keeps the last `MAX_SHOWN_PAPERS` (100) papers that aren't in the reading list, so Select buttons on much older cards stop responding.

## Accounts and chat history

- **Login:** `gui_login` checks accounts in `history.db` (table `accounts`, scrypt hashes). It never creates an account.
  - **Sign-up:** `/signup` creates one. Usernames are `[a-z0-9._-]{3,32}`, passwords at least 6 characters. Names that already own chats (e.g. `local`) are reserved.
  - **Admin:** `python db.py users | adduser NAME | deluser NAME`.
- **Login cookie:** kept for about 15 days (`user_session_timeout` in `.chainlit/config.toml`).
- **Per-user access:** each user only sees their own chats. `CleanupDataLayer` also blocks writes to another user's chat (`[owner] blocked write` in the log).
- **Deleting a chat** removes its rows from `history.db` and its files: `selections/selected_chunks_<chat id>.jsonl` and the `.files/<session id>/` folders recorded by `record_trace()`. Only files inside those two folders are deleted.
- **Preferences:** user preferences (Newest first) are in the `prefs` table.
- **Growth:** `history.db` keeps growing. For hosting, point `db.DATABASE_URL` at a hosted Postgres.
