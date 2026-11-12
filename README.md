# arXiv Paper Assistant (Chainlit)

A web app for finding computer science papers on arXiv (2007 to Oct 2026), collecting them in a reading list and exporting them for Zotero or Mendeley. It runs locally: a Qdrant vector database holds the papers, Ollama runs the language model, and Chainlit serves the web page. Several people can use it at the same time from other computers on the same network, each with their own login and chats.

## Using the app

Every message runs in a **mode**. The coloured banner at the top of the page shows the current mode and two example messages.

| Mode | Key | What it does | Example |
|---|---|---|---|
| Search (blue) | `search=` | Finds research papers and surveys on a topic | `search= generative AI for construction industry year=2020-` |
| Title (yellow) | `title=` | Finds one paper by its exact title | `title= Attention is all you need` |
| Analyze (orange) | `analyze=` (or `analyse=`) | Answers a question about papers already shown, using only their abstracts | `analyze= compare 1 and 2` |

- **The mode stays on.** After you type a key, your next messages need no key until you type another one. A new chat starts in search mode. A reopened chat keeps the mode it was in; the mode only changes when you type a key.
- **Filters** (search and title only; they apply to the message they're typed in):
  - `year=` takes `2021` (exact), `2019-2022` (range), `2019-` (from) or `-2020` (until).
  - `author=` takes one or more names separated by `;`, e.g. `author=Kipf; Welling`. With several names, a paper must have **all** of them.
  - In search mode they are strict filters. In title mode they only nudge the score.
- **Analyze references:** `1`, `[2]`, `S1` or `[S3]` are the labels of the **last** result. An arXiv id (e.g. `1706.03762`) can name any paper shown in the chat. Without a label, "this paper" or "it" means the best match; otherwise the app asks you which paper you mean.
- **Reading list:** click **☐ Select** under a paper to add it. The panel on the right lets you reorder and remove papers, and exports a zip with a BibTeX file (for Zotero/Mendeley), the PDFs and the selected text chunks.
- **Newest first** (checkbox in the top bar): shows papers and surveys by year instead of by relevance. It's saved per user.
- **Diagnostics:** the collapsible panel under each answer shows how the message was read, how candidates were ranked and how much of the model's context was used. It's saved with the chat.

## How a search works

1. **Input checks** (no model): message length (at most 200 words, any mode), keys, then repeated symbols are collapsed and other symbols removed.
2. **Spell check:** one Ollama call fixes the spelling of the topic or title.
3. **Retrieval:** Qdrant hybrid search (dense `BAAI/bge-base-en-v1.5` + BM25, merged with reciprocal rank fusion), once for the topic and once for "survey of <topic>". Results are split into research papers and surveys (by "survey" in the title).
4. **Reranking:** the `BAAI/bge-reranker-base` cross-encoder rescores both pools.
5. **Picking:**
   - Research papers: Ollama picks the best 3 of the reranker's top 5.
   - Surveys: the reranker's top 5 are shown directly, with no model call.

A title lookup skips step 5: it combines title similarity and the reranker score and returns one paper.

## Setup

You need:
- **Python 3.10+** (developed on 3.13)
- **[Ollama](https://ollama.com)** with a model pulled, e.g. `ollama pull qwen3:4b`
- **A Qdrant server** with the paper collection `sebun_bge768_hybrid_full`, built with `utils/db_generator_hybrid_gpu.py`. Run `python utils/db_generator_hybrid_gpu.py --help` for its options.
- **A GPU** is optional but makes the embedding and reranker models much faster.

Install the packages:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a `.env` file in the project folder. It is not in git, so never commit it:

```
CHAINLIT_AUTH_SECRET="<output of: .venv\Scripts\chainlit.exe create-secret>"
OLLAMA_MODEL="qwen3:4b"
QDRANT_URL="http://localhost:6333"
```

Optional settings, with their defaults:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama runs |
| `OLLAMA_NUM_CTX` | `16384` | Model context size in tokens |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded (`-1` = forever) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder used for reranking |
| `ANALYZE_MAX_TOKENS` | `2048` | Token cap for an analyze answer (thinking + answer) |

Changing `CHAINLIT_AUTH_SECRET` logs everyone out.

## Accounts

People can create their own account with **Create an account** on the login page. Usernames are 3–32 characters (`a-z 0-9 . _ -`) and passwords are at least 6 characters. Passwords are stored hashed (scrypt) in `history.db`.

Admin commands:

```powershell
python db.py users            # list accounts
python db.py adduser NAME     # create an account or reset its password (asks for the password)
python db.py deluser NAME     # remove an account (its chats are kept)
```

Each user only sees their own chats.

## Run it

```powershell
.venv\Scripts\python.exe -m chainlit run threeofus_main.py -w -h --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000.
- `-w` reloads the app whenever a `.py` file in the project folder is saved (all project modules are reloaded, not just `threeofus_main.py`), so no restart is needed after code changes.
- `-h` stops Chainlit from opening a browser window itself.
- `--host 0.0.0.0` makes the app reachable from other computers on the network at `http://<this-PC's-IP>:8000`.

On Windows, other computers can only connect once the firewall allows the port. Run this once in an admin PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Chainlit 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## Files

| File | What it does |
|---|---|
| `threeofus_main.py` | Main app: input checks, modes, search pipeline, LLM calls, Chainlit handlers, sign-up and preference routes |
| `notes.md` | Developer notes: how each step of the pipeline works, chat resume, accounts |
| `prompt_cleaning.py` | Input checks before any model call: key parser (modes, `year=`, `author=`), length limits, symbol cleaning |
| `retriever_hybrid_rrf.py` | Qdrant hybrid search (dense + BM25, RRF) and metadata filters |
| `chatbot_diagnostics.py` | Diagnostics: tagged notes, rank trace per candidate, context usage per LLM call |
| `app.py` | GUI helpers: paper cards, Select buttons, reading-list panel, mode banner messages, chat resume |
| `db.py` | `history.db` (SQLite): chats, accounts, preferences; per-user access checks; account commands |
| `export.py` | Builds the reading-list zip: BibTeX, PDFs from arXiv, selected chunks |
| `public/custom.js` | Page tweaks: mode banner, Newest first checkbox, sign-up link, reading-list tab, toasts |
| `public/elements/ReadingList.jsx` | Reading-list panel on the right |
| `public/elements/Diagnostics.jsx` | Diagnostics panel under each answer |
| `.chainlit/config.toml` | Chainlit settings (bump the `?v=` after `custom.js` when you change that file, so browsers reload it) |
| `utils/db_generator_hybrid_gpu.py` | Builds the Qdrant collection from arXiv metadata |
| `rag.py`, `chat.py`, `chainlit_app.py`, `data/` | Earlier prototype, not used by the current app |

## Known limits

- **Analyze answers are only as good as the abstracts.** Questions about datasets or exact numbers often get "the abstracts don't say".
- **A plain number in an analyze question counts as a paper label.** In "what are the top 3 ideas", `3` is read as paper 3. The answer always lists the papers it used.
- **Surveys are detected by the word "survey" in the title**, so reviews, overviews and tutorials count as research papers.
- **Exported zips are not saved with old chats.** In a reopened chat, export again.
- **`history.db` is a local file.** For hosting on a server that wipes its disk on restart, point `DATABASE_URL` in `db.py` to a hosted Postgres database instead.
