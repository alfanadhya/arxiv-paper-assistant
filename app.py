"""Chainlit prototype: ask -> top 5 papers -> add to reading list -> export zip.

Run from the project folder:
    .venv/Scripts/chainlit.exe run app.py -w
Then open http://localhost:8000

How the pieces talk to each other:
    user question ─► on_message ─► rag.ask() ─► 5 paper messages, each with an "Add" button
    "Add" button  ─► add_paper callback ─► reading list in user_session ─► refresh sidebar
    sidebar (public/elements/ReadingList.jsx):
        Bin button    ─► callAction("remove_paper") ─► remove_paper callback
        Export button ─► callAction("export")       ─► export callback ─► zip in chat

Chat history (left panel):
    login ─► password_auth_callback ─► Chainlit saves every chat in history.db (db.py)
    click an old chat ─► on_chat_resume ─► Chainlit restores user_session
                         (reading_list, history) ─► we redraw the reading-list panel
"""

import json
import os

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

import db
import export
import rag

MAX_HISTORY_TURNS = 10  # how many past question/answer pairs to keep for LangChain memory later


# ---------------------------------------------------------------- login + database
# The left history panel only appears when BOTH of these are set up:
# a login (so Chainlit knows whose chats to list) and a data layer (where chats are saved).
@cl.password_auth_callback
def login(username: str, password: str):
    # Prototype: anyone can log in with any username + the shared password from .env.
    # Each username gets its own chat history.
    if password == os.environ["DEMO_PASSWORD"] and username.strip():
        return cl.User(identifier=username.strip().lower())
    return None


@cl.data_layer
def get_data_layer():
    db.create_tables()
    return SQLAlchemyDataLayer(conninfo=db.DATABASE_URL)


# ---------------------------------------------------------------- helpers
def reading_list() -> list[dict]:
    return cl.user_session.get("reading_list", [])


async def refresh_sidebar():
    """Redraw the right-hand panel with the current reading list."""
    await cl.ElementSidebar.set_title("📚 Reading List")
    await cl.ElementSidebar.set_elements([
        cl.CustomElement(name="ReadingList", props={"papers": reading_list()}),
    ])


def remember(role: str, content: str):
    """Keep the conversation as a plain list of {"role", "content"} dicts.

    Chainlit saves user_session into the database with the chat, so this list
    comes back when the user reopens an old chat from the left panel.
    Later, LangChain can use it directly: a ChatPromptTemplate with
    MessagesPlaceholder("history") accepts dicts in this format.
    """
    history = cl.user_session.get("history", [])
    history.append({"role": role, "content": content})
    cl.user_session.set("history", history[-2 * MAX_HISTORY_TURNS:])


# Buttons. Chainlit saves messages in the database but NOT their buttons, so
# each message also stores what its button needs in `metadata` (which IS saved).
# When an old chat is reopened, resume() rebuilds the buttons from that metadata.
def add_button(arxiv_id: str) -> cl.Action:
    return cl.Action(name="add_paper", payload={"arxiv_id": arxiv_id},
                     label="Add to reading list", icon="plus")


def open_list_button() -> cl.Action:
    return cl.Action(name="open_list", payload={}, label="Reading list", icon="book-open")


def paper_card(p: dict) -> str:
    authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
    return (
        f"### [{p['title']}]({p['abs_url']})\n"
        f"`{p['arxiv_id']}` · {p['year']} · {authors}\n\n"
        f"_{p['why_relevant']}_\n\n"
        f"{p['abstract_snippet']}"
    )


# ---------------------------------------------------------------- chat
@cl.on_chat_start
async def start():
    cl.user_session.set("reading_list", [])
    cl.user_session.set("papers_by_id", {})   # every paper shown so far, so buttons only need the ID
    cl.user_session.set("history", [])
    await cl.Message(
        "Hi! Ask about a research topic and I'll show the 5 most related arXiv papers. "
        "Click **➕ Add** to put a paper in your reading list, then export it as a zip "
        "(BibTeX for Zotero/Mendeley + PDFs)."
    ).send()


@cl.on_chat_resume
async def resume(thread: dict):
    # Chainlit has already restored user_session (reading_list, papers_by_id, history)
    # and the messages from the database. Buttons and the panel aren't saved, so
    # rebuild them: find each message's metadata and attach its button again.
    for step in thread.get("steps", []):
        meta = step.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        if "arxiv_id" in meta:
            await add_button(meta["arxiv_id"]).send(for_id=step["id"])
        elif meta.get("kind") == "summary":
            await open_list_button().send(for_id=step["id"])
    if reading_list():
        await refresh_sidebar()


@cl.on_message
async def on_message(msg: cl.Message):
    # history lets the real RAG understand follow-ups like "more papers like the second one".
    result = rag.ask(msg.content, history=cl.user_session.get("history"))
    await cl.Message(result["summary"], actions=[open_list_button()],
                     metadata={"kind": "summary"}).send()

    remember("user", msg.content)
    # Save the titles too, so a follow-up question can refer to "the second paper".
    remember("assistant", result["summary"] + "\n" + "\n".join(
        f"{i}. [{p['arxiv_id']}] {p['title']}" for i, p in enumerate(result["papers"], 1)))

    shown = cl.user_session.get("papers_by_id")
    for p in result["papers"]:
        shown[p["arxiv_id"]] = p
        await cl.Message(content=paper_card(p), actions=[add_button(p["arxiv_id"])],
                         metadata={"arxiv_id": p["arxiv_id"]}).send()


# ---------------------------------------------------------------- buttons
@cl.action_callback("open_list")
async def open_list(action: cl.Action):
    await refresh_sidebar()


@cl.action_callback("add_paper")
async def add_paper(action: cl.Action):
    paper = cl.user_session.get("papers_by_id")[action.payload["arxiv_id"]]
    items = reading_list()
    if any(p["arxiv_id"] == paper["arxiv_id"] for p in items):
        await cl.context.emitter.send_toast("Already in your reading list", "info")
    else:
        items.append(paper)
        cl.user_session.set("reading_list", items)
        await cl.context.emitter.send_toast(f"Added: {paper['title'][:60]}", "success")
    await refresh_sidebar()


@cl.action_callback("remove_paper")
async def remove_paper(action: cl.Action):
    items = [p for p in reading_list() if p["arxiv_id"] != action.payload["arxiv_id"]]
    cl.user_session.set("reading_list", items)
    await refresh_sidebar()


@cl.action_callback("reorder")
async def reorder(action: cl.Action):
    """The panel sends the new order as a list of IDs, e.g. ["2301.11709", "2306.16313", ...]."""
    by_id = {p["arxiv_id"]: p for p in reading_list()}
    new_order = [by_id[i] for i in action.payload["order"] if i in by_id]
    # Safety net: keep any paper the panel didn't mention, at the end.
    new_order += [p for p in reading_list() if p["arxiv_id"] not in action.payload["order"]]
    cl.user_session.set("reading_list", new_order)
    await refresh_sidebar()


@cl.action_callback("export")
async def export_list(action: cl.Action):
    items = reading_list()
    if not items:
        await cl.context.emitter.send_toast("Your reading list is empty", "warning")
        return

    status = cl.Message(f"Preparing {len(items)} papers. arXiv allows one download every "
                        f"3 seconds, so this takes about {3 * len(items)} s...")
    await status.send()

    async def on_progress(i, n, p):
        status.content = f"Downloading PDF {i}/{n}: `{p['arxiv_id']}` {p['title'][:60]}"
        await status.update()

    data, failed = await export.build_zip(items, on_progress)

    status.content = f"✅ Done: {len(items)} papers. Import `reading_list.bib` into Zotero or Mendeley."
    if failed:
        status.content += f"\n\n⚠️ These PDFs failed to download: {', '.join(failed)}"
    status.elements = [cl.File(name="reading_list.zip", content=data, display="inline",
                               mime="application/zip")]
    await status.update()

    # Keep the list after exporting (the user may add more papers and export again),
    # and draw the panel again so it stays open.
    await refresh_sidebar()
