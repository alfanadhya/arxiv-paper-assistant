"""Chainlit GUI helpers (paper cards, select buttons, reading-list sidebar, zip export, chat resume).

No handlers are registered here: threeofus_main.py owns the @cl.on_* / @cl.action_callback
decorators (and the auto-login + data layer) and calls these functions.
The sidebar needs public/elements/ReadingList.jsx; saved chats live in history.db (db.py).
"""

import json

import chainlit as cl

import export

MAX_HISTORY_TURNS = 10  # question/answer pairs kept for follow-up questions
MAX_SHOWN_PAPERS = 100  # papers kept in papers_by_id (session is saved to the DB, capped at 1 MB)

WELCOME = ("Hi! Type a topic to search the arXiv papers. The banner at the top shows the current mode "
           "with examples; switch with **search=**, **title=** or **analyze=**. Optional filters: **year=** "
           "(*2021*, *2019-2022*, *2019-* or *-2020*) and **author=** (several separated by *;*, e.g. "
           "*author=Kipf; Welling*). Click **☐ Select** under a paper to add it to your reading list.")


# ---------------------------------------------------------------- session state
# Current reading list (papers bearing a checkmark, in user order).
def reading_list() -> list[dict]:
    return cl.user_session.get("reading_list", [])


# Paper dict shown earlier in this chat, by arXiv id.
def paper_by_id(arxiv_id: str) -> dict | None:
    return cl.user_session.get("papers_by_id", {}).get(arxiv_id)


# Reset the GUI state for a new chat.
def init_session() -> None:
    cl.user_session.set("reading_list", [])
    cl.user_session.set("papers_by_id", {})   # every paper shown so far
    cl.user_session.set("buttons", {})        # arxiv_id -> (select Action, message id)
    cl.user_session.set("history", [])        # conversation memory for follow-ups
    cl.user_session.set("sidebar_pinned", True)   # reading-list panel pinned by default
    cl.user_session.set("mode", "search")     # remembered mode: "search" / "title" / "analyze" (banner); new chat: search
    cl.user_session.set("last_results", [])   # [[label, arxiv_id], ...] of the last result, for analyze=


# After a resume, add any session keys that were not restored (keeps restored ones).
def ensure_session() -> None:
    for key, empty in (("reading_list", []), ("papers_by_id", {}), ("history", []), ("last_results", [])):
        if not isinstance(cl.user_session.get(key), type(empty)):
            cl.user_session.set(key, empty)
    if cl.user_session.get("mode") not in ("search", "title", "analyze"):
        cl.user_session.set("mode", None)
    cl.user_session.set("buttons", {})        # Action objects are never restored


# ---------------------------------------------------------------- mode (banner) and last results
# Remembered mode of this chat (None until the first search= / title= / analyze=).
def current_mode() -> str | None:
    return cl.user_session.get("mode")


# Remember the mode and show it in the top-bar banner (public/custom.js, section 6).
async def set_mode(mode: str | None) -> None:
    cl.user_session.set("mode", mode)
    await cl.send_window_message({"type": "mode", "mode": mode})


# Remember the labels of a result just shown (replaces the previous one), for analyze= references.
def set_last_results(labelled_hits: list[tuple[str, object]]) -> None:
    cl.user_session.set("last_results", [[label, hit.arxiv_id] for label, hit in labelled_hits])


# Papers of the last result in display order, each a paper dict with its label in that result.
def last_result_papers() -> list[dict]:
    papers = []
    for label, arxiv_id in cl.user_session.get("last_results") or []:
        paper = paper_by_id(arxiv_id)
        if paper is not None:
            papers.append({**paper, "label": label})
    return papers


# ---------------------------------------------------------------- conversation memory
# Conversation so far as [{"role", "content"}] (LangChain MessagesPlaceholder format).
def history() -> list[dict]:
    return cl.user_session.get("history", [])


# Append one turn to the session history, keeping the last MAX_HISTORY_TURNS pairs.
def remember(role: str, content: str) -> None:
    items = history()
    items.append({"role": role, "content": content})
    cl.user_session.set("history", items[-2 * MAX_HISTORY_TURNS:])


# ---------------------------------------------------------------- result display (data -> GUI)
# These only reshape search results for display; they take plain values, so app.py never imports
# threeofus_main (which would load a second copy of the module under Chainlit).

# Publication year from a hit's metadata; 0 if missing.
def paper_year(hit) -> int:
    year = (getattr(hit, "metadata", None) or {}).get("year")
    return year if isinstance(year, int) else 0


# Hits picked by the model, in display order, with their labels ("3", "S2"); r is a SearchResponse.
def picked_hits(r) -> list[tuple[str, object]]:
    papers = [(str(n), r.papers[n - 1]) for n in r.top_papers]
    surveys = [(f"S{n}", r.surveys[n - 1]) for n in r.top_surveys]
    return papers + surveys


# 'Last, First[, Suffix]' -> 'First Last[ Suffix]' (BibTeX export needs the last name last).
def display_author(name: str) -> str:
    parts = [p.strip() for p in name.split(",")]
    return " ".join([parts[1], parts[0]] + parts[2:]) if len(parts) > 1 and parts[1] else parts[0]


# Convert a retrieved hit into the paper dict used by the GUI and the zip export.
def to_paper(hit, label: str, why: str = "") -> dict:
    meta = hit.metadata
    authors = meta.get("authors") or []
    if isinstance(authors, str):
        authors = authors.split("; ")
    categories = meta.get("categories") or [meta.get("primary_category") or "cs"]
    if isinstance(categories, str):
        categories = categories.split()
    abstract = hit.text.split("\n\n", 1)[-1]
    return {
        "chunk_id": hit.point_id, "label": label, "arxiv_id": hit.arxiv_id, "title": hit.title,
        "authors": [display_author(a) for a in authors], "year": paper_year(hit), "categories": categories,
        "abstract": abstract, "abstract_snippet": abstract[:300] + ("..." if len(abstract) > 300 else ""),
        "why_relevant": why, "score": round(hit.vector_score, 4),
        "abs_url": hit.url, "pdf_url": f"https://arxiv.org/pdf/{hit.arxiv_id}",
        # compact copy of the chunk metadata, so a resumed chat can still write selected chunks
        "metadata": {k: meta[k] for k in ("year", "published", "primary_category", "categories",
                                          "authors", "doi", "journal_ref") if k in meta},
    }


# Rebuild a minimal hit from a saved paper dict (used after a chat is resumed from the DB).
def paper_to_hit(p: dict):
    from types import SimpleNamespace
    return SimpleNamespace(point_id=p["chunk_id"], arxiv_id=p["arxiv_id"], title=p["title"], url=p["abs_url"],
                           text=f"{p['title']}\n\n{p['abstract']}", vector_score=p.get("score", 0.0),
                           metadata=p.get("metadata") or {"year": p.get("year")})


# Markdown summary of a successful search response; filters_text comes from describe_filters().
def summary_md(r, filters_text: str) -> str:
    lines = []
    if r.search_prompt and r.search_prompt != r.prompt:
        lines.append(f"**Follow-up understood as:** {r.search_prompt}")
    label = "Title" if getattr(r, "mode", "search") == "title" else "Topic"
    lines.append(f"**{label}:** {r.analysis.corrected_query} · **Filters:** {filters_text}")
    if r.analysis.authors and getattr(r, "mode", "search") == "title":   # search: shown in the Filters line
        lines.append(f"**Author hint (bonus, not a filter):** {', '.join(r.analysis.authors)}")
    lines += [f"⚠️ {w}" for w in r.warnings]
    if r.criteria:
        lines.append("**Evaluation criteria:**\n" + "\n".join(f"- {c}" for c in r.criteria))
    if r.best:
        lines.append(f"**Best match:** [{r.best[0]}] {r.best[1].title}\n\n_{r.why_best}_")
    if not picked_hits(r):
        lines.append("The model picked no papers from the results.")
    return "\n\n".join(lines)


# ---------------------------------------------------------------- widgets
# Buttons get a fixed id per message (not a random one): the browser appends every button it receives and
# removes only by id, so re-sending a message's button (chat resume after a reconnect/server restart) must
# first remove the copy the browser may still show; see resend().
def _button_id(kind: str, msg_id: str | None) -> str:
    import uuid
    return f"{kind}-{msg_id}" if msg_id else f"{kind}-{uuid.uuid4()}"


# Show a button under a message exactly once: drop a copy with the same id, then send it.
async def resend(action: cl.Action, for_id: str) -> None:
    await action.remove()
    await action.send(for_id=for_id)


# Select/unselect button for one paper; label shows the checkmark state.
def select_button(arxiv_id: str, checked: bool, msg_id: str | None = None) -> cl.Action:
    return cl.Action(name="add_paper", payload={"arxiv_id": arxiv_id},
                     label="☑ Selected" if checked else "☐ Select",
                     tooltip="Click to remove from reading list" if checked else "Click to add to reading list",
                     id=_button_id("select", msg_id))


# Original "Add to reading list" button (kept; select_button replaced it in the chat).
def add_button(arxiv_id: str) -> cl.Action:
    return cl.Action(name="add_paper", payload={"arxiv_id": arxiv_id},
                     label="Add to reading list", icon="plus")


# Button that opens the reading-list sidebar. Its id starts with 'open-list-' so the always-visible
# "Reading list" tab (public/custom.js) can find and click one of these buttons.
def open_list_button(msg_id: str | None = None) -> cl.Action:
    return cl.Action(name="open_list", payload={}, label="Reading list", icon="book-open",
                     id=_button_id("open-list", msg_id))


# Markdown card for one paper.
def paper_card(p: dict) -> str:
    authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
    why = f"_{p['why_relevant']}_\n\n" if p.get("why_relevant") else ""
    # Only the best match has a "why"; its link gets a '#best' marker, which public/custom.js colours green
    url = p["abs_url"] + ("#best" if p.get("why_relevant") else "")
    return (f"### [{p['label']}] [{p['title']}]({url})\n"
            f"`{p['arxiv_id']}` · {p['year'] or 'year unknown'} · {authors}\n\n"
            f"{why}{p['abstract_snippet']}")


# True if the reading-list panel is pinned (default).
def is_pinned() -> bool:
    return cl.user_session.get("sidebar_pinned", True) is not False


# Save the pin state sent by the panel.
def set_pinned(pinned: bool) -> None:
    cl.user_session.set("sidebar_pinned", bool(pinned))


# Redraw (and open) the right-hand panel with the current reading list.
# A new key/version each time: Chainlit ignores set_elements with an unchanged key, and the panel uses
# the version to decide whether to keep its own (browser-side) state or take the server's list.
async def refresh_sidebar() -> None:
    version = cl.user_session.get("sidebar_version", 0) + 1
    cl.user_session.set("sidebar_version", version)
    await cl.ElementSidebar.set_title("📚 Reading List")
    await cl.ElementSidebar.set_elements([
        cl.CustomElement(name="ReadingList",
                         props={"papers": reading_list(), "pinned": is_pinned(), "version": version}),
    ], key=f"reading-list-{version}")


# Send the current reading list to the page without redrawing the panel (no reload): an open panel updates
# its list in place (ReadingList.jsx); a closed one is opened by public/custom.js.
async def push_reading_list() -> None:
    import time
    await cl.send_window_message({"type": "reading_list", "papers": reading_list(), "pinned": is_pinned(),
                                  "seq": time.time_ns()})   # seq: the panel confirms it applied this update


# Close the right-hand panel.
async def close_sidebar() -> None:
    await cl.ElementSidebar.set_elements([])


# ---------------------------------------------------------------- messages
# Greeting shown when a chat starts; it carries a "Reading list" button, so the always-visible tab
# always has a button to trigger, even before the first search.
async def send_welcome() -> None:
    msg = cl.Message(WELCOME)
    msg.actions = [open_list_button(msg.id)]
    await msg.send()


# Model summary message with a "Reading list" button.
async def send_summary(summary: str) -> None:
    msg = cl.Message(summary, metadata={"kind": "summary"})
    msg.actions = [open_list_button(msg.id)]
    await msg.send()


# One paper card with its select button; remembers the paper and button.
async def send_paper(p: dict, checked: bool) -> None:
    shown = cl.user_session.get("papers_by_id")
    shown.pop(p["arxiv_id"], None)
    shown[p["arxiv_id"]] = p
    keep = {q["arxiv_id"] for q in reading_list()}
    for old_id in [i for i in shown if i not in keep][:-MAX_SHOWN_PAPERS]:   # drop oldest unselected
        del shown[old_id]
    msg = cl.Message(content=paper_card(p), metadata={"arxiv_id": p["arxiv_id"]})
    button = select_button(p["arxiv_id"], checked, msg.id)
    msg.actions = [button]
    await msg.send()
    cl.user_session.get("buttons")[p["arxiv_id"]] = (button, msg.id)


# Plain error/notice message.
async def send_notice(text: str) -> None:
    await cl.Message(text).send()


# Diagnostics panel element for a message; a fixed id per message so a resume never shows it twice.
def diagnostics_panel(data: dict, msg_id: str) -> cl.CustomElement:
    return cl.CustomElement(name="Diagnostics", props={**data, "panel_id": msg_id}, display="inline",
                            id=f"diag-{msg_id}")


# Collapsible diagnostics panel (public/elements/Diagnostics.jsx) shown under the answer. The data is also
# saved in the message's metadata (steps.metadata in history.db), so resume() can redraw it with the chat.
async def send_diagnostics_panel(data: dict) -> None:
    msg = cl.Message(content="", author="diagnostics", metadata={"kind": "diagnostics", "diagnostics": data})
    msg.elements = [diagnostics_panel(data, msg.id)]
    await msg.send()


# Diagnostic digest as a plain-text message (kept; the GUI now uses send_diagnostics_panel).
async def send_diagnostics(lines: list[str]) -> None:
    if lines:
        await cl.Message("**Diagnostic digest**\n" + "\n".join(f"- {d}" for d in lines),
                         author="diagnostics").send()


# Swap a paper's select button so its label matches the checkmark state.
async def set_button_state(arxiv_id: str, checked: bool) -> None:
    buttons = cl.user_session.get("buttons")
    if arxiv_id not in buttons:
        return
    old, msg_id = buttons[arxiv_id]
    await old.remove()
    new = select_button(arxiv_id, checked, msg_id)
    await resend(new, msg_id)
    buttons[arxiv_id] = (new, msg_id)


# Rebuild buttons and the sidebar when an old chat is reopened (call ensure_session() first).
async def resume(thread: dict) -> None:
    selected = {p["arxiv_id"] for p in reading_list()}
    buttons = cl.user_session.get("buttons")
    steps = thread.get("steps", [])
    has_list_button = False
    for step in steps:
        meta = step.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        if "arxiv_id" in meta and paper_by_id(meta["arxiv_id"]) is not None:
            button = select_button(meta["arxiv_id"], meta["arxiv_id"] in selected, step["id"])
            await resend(button, step["id"])   # never a 2nd copy if the browser still shows the old one
            buttons[meta["arxiv_id"]] = (button, step["id"])
        elif meta.get("kind") == "summary":
            await resend(open_list_button(step["id"]), step["id"])
            has_list_button = True
        elif meta.get("kind") == "diagnostics" and meta.get("diagnostics"):
            # Redraw the saved diagnostics panel under its own message. Chainlit sends `thread` to the browser
            # AFTER this resume and the browser then replaces all its elements with thread["elements"] (empty:
            # elements are not stored), so a panel sent separately would vanish. Adding it to that list keeps it.
            panel = diagnostics_panel(meta["diagnostics"], step["id"])
            panel.for_id = step["id"]
            if not isinstance(thread.get("elements"), list):
                thread["elements"] = []
            thread["elements"].append(panel.to_dict())
    if not has_list_button and steps:   # keep a button for the always-visible "Reading list" tab
        await resend(open_list_button(steps[0]["id"]), steps[0]["id"])
    if reading_list():
        await refresh_sidebar()
    # A reopened chat keeps the mode it was in (the mode only changes when the user types a mode key).
    # Default only for chats saved before modes existed: analyze if it has papers, else search.
    mode = current_mode() or ("analyze" if cl.user_session.get("papers_by_id") else "search")
    await set_mode(mode)


# ---------------------------------------------------------------- reading list
# Add a shown paper to the reading list; returns False if already there (no toast: the card button shows it).
async def add_to_list(arxiv_id: str) -> bool:
    paper, items = paper_by_id(arxiv_id), reading_list()
    if paper is None or any(p["arxiv_id"] == arxiv_id for p in items):
        return False
    items.append(paper)
    cl.user_session.set("reading_list", items)
    return True


# Remove a paper from the reading list; returns False if it was not there.
async def remove_from_list(arxiv_id: str) -> bool:
    items = reading_list()
    kept = [p for p in items if p["arxiv_id"] != arxiv_id]
    cl.user_session.set("reading_list", kept)
    return len(kept) != len(items)


# Apply the order sent by the sidebar (list of arXiv ids); unknown ids are kept at the end.
def reorder(order: list[str]) -> None:
    by_id = {p["arxiv_id"]: p for p in reading_list()}
    new_order = [by_id[i] for i in order if i in by_id]
    new_order += [p for p in reading_list() if p["arxiv_id"] not in order]
    cl.user_session.set("reading_list", new_order)


# Build and send the zip (BibTeX + PDFs) for the reading list; returns False if empty.
async def export_list() -> bool:
    items = reading_list()
    if not items:
        await send_notice("Your reading list is empty: click **☐ Select** under a paper first.")
        return False

    status = cl.Message(f"Preparing {len(items)} papers. arXiv allows one download every "
                        f"3 seconds, so this takes about {3 * len(items)} s...")
    await status.send()

    # Show download progress in the status message.
    async def on_progress(i, n, p):
        status.content = f"Downloading PDF {i}/{n}: `{p['arxiv_id']}` {p['title'][:60]}"
        await status.update()

    try:
        data, failed = await export.build_zip(items, on_progress)
    except Exception as exc:
        status.content = f"❌ Export failed: {type(exc).__name__}: {exc}"
        await status.update()
        return False
    status.content = (f"✅ Done: {len(items)} papers. Click **reading_list.zip** below to download, "
                      "then import `reading_list.bib` into Zotero or Mendeley.")
    if failed:
        status.content += f"\n\n⚠️ These PDFs failed to download: {', '.join(failed)}"
    await status.update()
    # Message.update() does not send new elements, so attach the zip to the message explicitly
    zip_file = cl.File(name="reading_list.zip", content=data, display="inline", mime="application/zip")
    await zip_file.send(for_id=status.id)
    return True
