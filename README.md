# arXiv Paper Assistant

A web app for finding computer science papers on arXiv (2007 to Oct 2026), collecting them in a reading list and exporting them for Zotero or Mendeley. It runs locally: a Qdrant vector database holds the papers, Ollama runs the language model, and Chainlit serves the web page. Several people can use it at the same time from other computers on the same network, each with their own login and chats.

## Using the app

A new chat opens on the **Home** screen: the app's goal, the three ways to ask (Research, Search, Title) with example messages to click, and short tips.

Type a question in your own words and the **research assistant** answers it: the model searches the papers itself and cites the ones it used, each shown as a card under the answer.

Two **keywords** at the start of a message skip the assistant and run a fixed search instead. A keyword applies only to the message it is typed in:

| Keyword | What it does | Example |
|---|---|---|
| `search=` | Lists research papers and surveys on a topic | `search= generative AI for construction industry year=2020-` |
| `title=` | Finds one paper by its exact title | `title= Attention is all you need` |

- **Filters** (with `search=` or `title=`):
  - `year=` takes `2021` (exact), `2019-2022` (range), `2019-` (from) or `-2020` (until).
  - `author=` takes one or more names separated by `;`, e.g. `author=Kipf; Welling`. With several names, a paper must have **all** of them.
  - With `search=` they are strict filters. With `title=` they only nudge the score.
- **Numbers and follow-ups:** the papers of each result are numbered from 1, and numbers always refer to the latest result ("compare 1 and 2", "list all authors of 1 and 3"). A question about the papers on screen keeps their numbers.
- **Reading list:** click **☐ Select** under a paper to add it. The panel on the right lets you reorder and remove papers, and exports a zip with a BibTeX file (for Zotero/Mendeley), the PDFs and the selected text chunks.
- **Newest first** (checkbox in the top bar): shows the papers of a `search=` by year instead of by relevance. It's saved per user.
- **Settings** (⚙ in the top bar), saved per user: the reranker cutoff, and whether diagnostics panels open expanded.
- **Diagnostics:** the collapsible panel under each answer shows how the message was read, what was searched, how candidates were ranked and how much of the model's context was used. It's saved with the chat.

## How a search works

1. **Input checks** (no model): message length (at most 180 words), keywords, then repeated symbols are collapsed and other symbols removed.
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
| `OLLAMA_NUM_CTX` | `auto` | Model context size in tokens. `auto` = the model's own maximum, read from Ollama at start-up, but at most `NUM_CTX_MAX`; a number fixes it |
| `NUM_CTX_MAX` | `16384` | Upper limit for `auto` (Ollama reserves GPU memory for the whole window) |
| `RANKING_OUTPUT_RESERVE` | `2000` | Least output room a ranking call gets: if the prompt would leave less, fewer papers are sent; if even one paper leaves less, the reranker order is used |
| `RANKING_MAX_OUTPUT` | `0` | Each ranking call may write what the window has left after its prompt; a number above 0 also caps it at that many tokens |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded (`-1` = forever) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder used for reranking |

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
- **Windows with Python 3.14:** start with `.venv\Scripts\python.exe run_local.py` instead (same options). `chainlit run` there leaves a blank white page, see `run_local.py`.

On Windows, other computers can only connect once the firewall allows the port. Run this once in an admin PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Chainlit 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## Files

| File | What it does |
|---|---|
| `threeofus_main.py` | Main app: input checks, keywords, research assistant (agent), search pipeline, LLM calls, Chainlit handlers, sign-up and preference routes |
| `chainlit.md` | The readme shown inside the app (user guide) |
| `notes.md` | Developer notes: how each step of the pipeline works, chat resume, accounts |
| `prompt_cleaning.py` | Input checks before any model call: keyword parser (`search=`, `title=`, `year=`, `author=`), length limits, symbol cleaning |
| `retriever_hybrid_rrf.py` | Qdrant hybrid search (dense + BM25, RRF) and metadata filters |
| `chatbot_diagnostics.py` | Diagnostics: tagged notes, rank trace per candidate, context usage per LLM call |
| `app.py` | GUI helpers: Home screen message, paper cards, Select buttons, reading-list panel, per-message signal for the input border, chat resume |
| `db.py` | `history.db` (SQLite): chats, accounts, preferences; per-user access checks; account commands |
| `export.py` | Builds the reading-list zip: BibTeX, PDFs from arXiv, selected chunks |
| `public/custom.js` | Page tweaks: light theme, input border colour per message, plain chat list, Newest first checkbox, Settings button, sign-up link, reading-list tab, loading animation, toasts |
| `public/theme.json`, `public/theme.css` | Look of the app (light "Base" style: colours, fonts, Home screen, loading animation); bump `?v=` in config.toml after a change |
| `public/elements/Home.jsx` | Home screen shown when a chat starts |
| `public/avatars/` | Grey circle used as the assistant's avatar |
| `run_local.py` | Starts the app on a Windows PC with Python 3.14 (see Run it) |
| `public/elements/ReadingList.jsx` | Reading-list panel on the right |
| `public/elements/Diagnostics.jsx` | Diagnostics panel under each answer |
| `.chainlit/config.toml` | Chainlit settings (bump the `?v=` after `custom.js` when you change that file, so browsers reload it) |
| `utils/db_generator_hybrid_gpu.py` | Builds the Qdrant collection from arXiv metadata |
| `rag.py`, `chat.py`, `chainlit_app.py`, `data/` | Earlier prototype, not used by the current app |

## Known limits

- **Surveys are detected by the word "survey" in the title**, so reviews, overviews and tutorials count as research papers.
- **Exported zips are not saved with old chats.** In a reopened chat, export again.
- **`history.db` is a local file.** For hosting on a server that wipes its disk on restart, point `DATABASE_URL` in `db.py` to a hosted Postgres database instead.
