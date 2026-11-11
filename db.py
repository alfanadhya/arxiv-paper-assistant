"""The database that stores chat history (the left panel), as one SQLite file: history.db.

Chainlit's SQLAlchemyDataLayer writes users, threads (= chats), steps (= messages),
elements (= files) and feedbacks into these 5 tables. It doesn't create them itself,
so create_tables() does that once on startup. The column names must match what
Chainlit writes, which is why they're in camelCase with quotes.

SQLite is a single file, which is fine locally. For a hosted app whose disk gets
wiped on restart, switch DATABASE_URL to a free hosted Postgres (e.g. Neon/Supabase).

Check the file afterwards:  .venv/Scripts/python.exe -X utf8 db.py

Login accounts (table "accounts", salted scrypt hashes):
    python db.py adduser NAME     create an account / reset its password (asks for the password)
    python db.py deluser NAME     remove an account (its chats are kept)
    python db.py users            list accounts
"""

import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "history.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_FILE.as_posix()}"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT REFERENCES users("id") ON DELETE CASCADE,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL REFERENCES threads("id") ON DELETE CASCADE,
    "parentId" TEXT,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INTEGER,
    "defaultOpen" BOOLEAN,
    "autoCollapse" BOOLEAN,
    "icon" TEXT
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT REFERENCES threads("id") ON DELETE CASCADE,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL REFERENCES threads("id") ON DELETE CASCADE,
    "value" INTEGER NOT NULL,
    "comment" TEXT
);
-- Login accounts (ours, not Chainlit's): kept apart from "users" because Chainlit sends users.metadata
-- to the browser, and a password hash must never go there.
CREATE TABLE IF NOT EXISTS accounts (
    "username" TEXT PRIMARY KEY,
    "password_hash" TEXT NOT NULL,
    "createdAt" TEXT
);
-- Per-user GUI settings, e.g. the "Newest first" box in the top bar.
CREATE TABLE IF NOT EXISTS prefs (
    "username" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "value" TEXT,
    PRIMARY KEY ("username", "key")
);
"""


def create_tables() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- login accounts (salted scrypt hashes)
# Stored as "scrypt$n$r$p$<salt hex>$<hash hex>"; the plain password is never written anywhere.
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def hash_password(password: str) -> str:
    import hashlib
    import os
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    import hashlib
    import hmac
    try:
        _, n, r, p, salt, digest = stored.split("$")
        test = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(test.hex(), digest)
    except (ValueError, TypeError):
        return False


# Create the account, or reset its password if it already exists.
def set_account(username: str, password: str) -> str:
    from datetime import datetime, timezone
    name = normalize_username(username)
    if not name or not password:
        raise ValueError("username and password must not be empty")
    create_tables()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('INSERT INTO accounts ("username", "password_hash", "createdAt") VALUES (?, ?, ?) '
                     'ON CONFLICT("username") DO UPDATE SET "password_hash" = excluded."password_hash"',
                     (name, hash_password(password), datetime.now(timezone.utc).isoformat()))
    return name


# ---------------------------------------------------------------- per-user GUI settings
def get_pref(username: str, key: str, default: str | None = None) -> str | None:
    create_tables()
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute('SELECT "value" FROM prefs WHERE "username" = ? AND "key" = ?',
                           (normalize_username(username), key)).fetchone()
    return row[0] if row else default


def set_pref(username: str, key: str, value: str) -> None:
    create_tables()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('INSERT INTO prefs ("username", "key", "value") VALUES (?, ?, ?) '
                     'ON CONFLICT("username", "key") DO UPDATE SET "value" = excluded."value"',
                     (normalize_username(username), key, value))


USERNAME_RULE ="3-32 characters: lowercase letters, digits, '.', '_' or '-'"
MIN_PASSWORD = 6


# Self sign-up (the /signup page): create a NEW account, never change an existing one.
# Raises ValueError with a message for the user. A name that already owns chats but has no account
# (e.g. "local" from before the login existed) is reserved, so nobody can sign up and read those chats;
# the admin gives it a password with  python db.py adduser NAME.
def create_account(username: str, password: str) -> str:
    import re
    from datetime import datetime, timezone
    name = normalize_username(username)
    if not re.fullmatch(r"[a-z0-9._-]{3,32}", name):
        raise ValueError(f"Username must be {USERNAME_RULE}.")
    if len(password or "") < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
    create_tables()
    with sqlite3.connect(DB_FILE) as conn:
        if conn.execute('SELECT 1 FROM users WHERE "identifier" = ?', (name,)).fetchone() and \
                not conn.execute('SELECT 1 FROM accounts WHERE "username" = ?', (name,)).fetchone():
            raise ValueError("That username is reserved. Please choose another one.")
        # DO NOTHING on a duplicate, so two simultaneous sign-ups can't overwrite each other's password.
        created = conn.execute(
            'INSERT INTO accounts ("username", "password_hash", "createdAt") VALUES (?, ?, ?) '
            'ON CONFLICT("username") DO NOTHING',
            (name, hash_password(password), datetime.now(timezone.utc).isoformat())).rowcount > 0
    if not created:
        raise ValueError("That username is already taken.")
    return name


# Remove the login only; the user's chats stay in history.db (re-adding the name gives them back).
def delete_account(username: str) -> bool:
    with sqlite3.connect(DB_FILE) as conn:
        return conn.execute('DELETE FROM accounts WHERE "username" = ?',
                            (normalize_username(username),)).rowcount > 0


def list_accounts() -> list[tuple[str, str]]:
    create_tables()
    with sqlite3.connect(DB_FILE) as conn:
        return conn.execute('SELECT "username", "createdAt" FROM accounts ORDER BY "username"').fetchall()


# The normalized username if the password matches its account, else None.
def check_login(username: str, password: str) -> str | None:
    name = normalize_username(username)
    if not name or not password:
        return None
    create_tables()
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute('SELECT "password_hash" FROM accounts WHERE "username" = ?', (name,)).fetchone()
    return name if row and verify_password(password, row[0]) else None


# ---------------------------------------------------------------- delete a chat = delete all its traces
# Session keys (saved in the thread's metadata) where the app records what a chat created on disk.
TRACE_FILES_KEY = "trace_files"            # files written for this chat (e.g. selections/*.jsonl)
TRACE_SESSIONS_KEY = "trace_session_ids"   # Chainlit sessions that served this chat (.files/<session id>/)


# True if path is inside folder (never delete anything outside the app's own folders).
def _inside(path: Path, folder: Path) -> bool:
    try:
        return path.resolve().is_relative_to(folder.resolve())
    except (OSError, ValueError):
        return False


# Remove a chat's files: recorded files + selections/*<thread id>* + .files/<session id>/ folders.
def delete_chat_files(thread_id: str, files: list[str], session_ids: list[str]) -> list[str]:
    import shutil
    from chainlit.config import FILES_DIRECTORY
    from export import SELECTION_DIR

    removed = []
    candidates = [Path(f) for f in files] + list(SELECTION_DIR.glob(f"*{thread_id}*"))
    for path in candidates:
        if _inside(path, SELECTION_DIR) and path.is_file():
            path.unlink(missing_ok=True)
            removed.append(str(path))
    for sid in session_ids:
        folder = Path(FILES_DIRECTORY) / str(sid)
        if sid and _inside(folder, Path(FILES_DIRECTORY)) and folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(str(folder))
    return removed


# Data layer that also deletes a chat's files when the chat is deleted from the left pane.
def make_data_layer():
    import json
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    class CleanupDataLayer(SQLAlchemyDataLayer):
        # Chat ids deleted while the server runs: late saves for them are ignored, so a closing session
        # (Chainlit saves the session when it disconnects, possibly after the delete) can't re-create the chat.
        deleted_ids: set[str] = set()
        owners: dict[str, str] = {}   # thread id -> owner username (a chat's owner never changes)

        # Second line of defence (Chainlit already checks the owner on resume/open/rename/delete):
        # True if the chat belongs to someone other than the logged-in user of the current session,
        # so one user's session can never write into another user's chat.
        async def _foreign_thread(self, thread_id: str | None) -> bool:
            if not thread_id:
                return False
            try:
                from chainlit.context import context
                user = context.session.user
            except Exception:          # no session (e.g. a background save without context): nothing to compare
                return False
            if user is None:
                return False
            owner = self.owners.get(thread_id)
            if owner is None:
                try:
                    owner = await self.get_thread_author(thread_id)
                except ValueError:     # chat not saved yet: it will be created for this user
                    return False
                self.owners[thread_id] = owner
            if owner != user.identifier:
                print(f"[owner] blocked write by '{user.identifier}' to chat {thread_id} of '{owner}'", flush=True)
                return True
            return False

        # Save chat metadata unless the chat was deleted or belongs to another user
        # (update_thread inserts the row if it is missing).
        async def update_thread(self, thread_id: str, *args, **kwargs):
            if thread_id in self.deleted_ids:
                print(f"[delete] ignored late save for deleted chat {thread_id}", flush=True)
                return
            if await self._foreign_thread(thread_id):
                return
            return await super().update_thread(thread_id, *args, **kwargs)

        # Save a message unless its chat was deleted (would otherwise leave orphan rows) or is someone else's.
        async def create_step(self, step_dict, *args, **kwargs):
            if step_dict.get("threadId") in self.deleted_ids or await self._foreign_thread(step_dict.get("threadId")):
                return
            return await super().create_step(step_dict, *args, **kwargs)

        # Update a message unless its chat was deleted or is someone else's.
        async def update_step(self, step_dict, *args, **kwargs):
            if step_dict.get("threadId") in self.deleted_ids or await self._foreign_thread(step_dict.get("threadId")):
                return
            return await super().update_step(step_dict, *args, **kwargs)

        # Delete the chat's rows (Chainlit: thread, steps, elements, feedbacks) and then its files.
        async def delete_thread(self, thread_id: str):
            self.deleted_ids.add(thread_id)   # first, so saves racing with the delete are already blocked
            self.owners.pop(thread_id, None)
            meta = {}
            try:
                thread = await self.get_thread(thread_id)   # read the trace before the rows are gone
                meta = (thread or {}).get("metadata") or {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
            except Exception as exc:
                print(f"[delete] could not read metadata of chat {thread_id}: {exc}", flush=True)
            await super().delete_thread(thread_id)
            removed = delete_chat_files(thread_id, meta.get(TRACE_FILES_KEY) or [],
                                        meta.get(TRACE_SESSIONS_KEY) or [])
            print(f"[delete] chat {thread_id}: database rows removed; files removed: {removed or 'none'}",
                  flush=True)

    create_tables()
    return CleanupDataLayer(conninfo=DATABASE_URL)


def _account_cli(args: list[str]) -> None:
    from getpass import getpass
    cmd = args[0]
    if cmd == "users":
        rows = list_accounts()
        for name, created in rows:
            print(f"  {name:<20} created {created}")
        print(f"{len(rows)} account(s)")
    elif cmd == "adduser" and len(args) == 2:
        password = getpass(f"Password for {args[1]}: ")
        if password != getpass("Repeat password: "):
            sys.exit("Passwords do not match; nothing changed.")
        print(f"Saved account '{set_account(args[1], password)}'.")
    elif cmd == "deluser" and len(args) == 2:
        print("Account removed (chats kept)." if delete_account(args[1]) else "No such account.")
    else:
        sys.exit("Usage: python db.py [users | adduser NAME | deluser NAME]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        _account_cli(sys.argv[1:])
        sys.exit()
    create_tables()
    with sqlite3.connect(DB_FILE) as conn:
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            print(f"{table:<10} {count} rows")
        print("\nlatest chats:")
        for name, user, created in conn.execute(
                'SELECT "name", "userIdentifier", "createdAt" FROM threads ORDER BY "createdAt" DESC LIMIT 5'):
            print(f"  {created}  {user:<10} {name}")
