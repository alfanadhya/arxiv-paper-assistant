# arXiv Paper Assistant (Chainlit)

A clickable prototype of the flow: **ask → top 5 papers → add to reading list → export zip**.
The search is a simple word match over the 5,520-paper arXiv cs.CL sample in `data/` (arXiv metadata, CC0), with no embeddings and no LLM. That's on purpose: the prototype tests the UI flow, not the search quality.

| File | What it does |
|---|---|
| `rag.py` | Stand-in for the real RAG. Its `ask()` returns the **contract** (the JSON shape) the UI depends on |
| `app.py` | Chainlit app: paper messages with an Add button, callbacks for add/remove/export |
| `public/elements/ReadingList.jsx` | The Reading List panel in the right sidebar (Bin + Export buttons) |
| `export.py` | Builds `reading_list.zip`: `reading_list.bib` (Zotero/Mendeley import) + PDFs from arXiv, 3 s apart |
| `db.py` | Creates `history.db` (SQLite), where Chainlit saves every chat for the left history panel |
| `.env` | The login secret + the shared demo password (`paper2026`). Not committed to git |

## Run it

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\chainlit.exe run app.py -w
```

Open http://localhost:8000. `-w` reloads the app whenever you save a file.

On a new computer, `.env` won't exist because it isn't in git. Create it with:
```
CHAINLIT_AUTH_SECRET="<output of: .venv\Scripts\chainlit.exe create-secret>"
DEMO_PASSWORD="paper2026"
```

## What to click (test script)

0. Log in with any username (e.g. `nami`) and the password `paper2026`. Each username has its own history.
1. Ask `retrieval augmented generation evaluation`. You should see 1 summary message + 5 paper cards.
2. Click **Add to reading list** on 2 papers. The right panel opens and shows them, with a green toast for each.
3. Click Add on the same paper again. You should get the toast "Already in your reading list", with no duplicate.
4. Click the 🗑 on one paper. It disappears from the panel.
5. Ask a second question and add 1 paper from it. The list now combines papers from both questions.
6. Click **Export and download**. The message counts PDFs as they download, then shows `reading_list.zip`. **The panel should stay open with the same list.** Unzip the file and import the `.bib` into Zotero.
7. Click **New Chat** (top left) and ask something else. Open the left history panel: the first chat is listed, named after its first question.
8. Click the first chat. Its messages come back **with their Add buttons**, and so does its reading list in the right panel.
9. Close the right panel, then click **Reading list** under any answer summary. The panel opens again.
10. Check the database: `.venv\Scripts\python.exe -X utf8 db.py` lists the tables and the latest chats.

## Connecting the real RAG later

Change only `rag.ask()`: call the real pipeline and return the `papers` shape described at the top of `rag.py`. See [INTEGRATION.md](INTEGRATION.md). `app.py` and the JSX panel stay the same.

## LangChain memory later

`app.py` already keeps the conversation in `cl.user_session["history"]` as `[{"role": "user"|"assistant", "content": ...}]`, capped at the last 10 turns. Chainlit saves it with the chat, so it also comes back when an old chat is reopened. To use it:
1. In the prompt: `ChatPromptTemplate.from_messages([("system", ...), MessagesPlaceholder("history"), ("human", "{question}")])`
2. Call it with `chain.invoke({"history": history, "question": q})`. LangChain accepts these dicts as they are.
3. Main use: rewrite a follow-up ("more like the second one") into a standalone search query *before* retrieval. The assistant turns include the numbered titles and IDs, so "the second one" can be resolved.

## Known limits

- Chainlit's database saves messages but **not their buttons**. So each message stores what its button needs in `metadata`, and `resume()` in `app.py` re-attaches the buttons when an old chat is reopened. A new kind of button needs the same treatment.
- Each chat has **its own** reading list. A single list per user across all chats (more like Zotero) would need to be stored on the user instead. Decide which one you want.
- The reading list, history, etc. are saved to the database when the browser tab **disconnects** (closed, or switched to another chat). If the server crashes mid-chat, the latest changes are lost.
- Files (the exported zip) are **not** saved with old chats. That would need a storage provider such as S3. In a reopened chat, export again.
- `history.db` is a local file. Many free hosts wipe the disk on restart, so for deployment switch `DATABASE_URL` in `db.py` to a free hosted Postgres (Neon, Supabase). The table names stay the same.
