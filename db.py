"""The database that stores chat history (the left panel), login accounts and per-user settings.

Two backends, chosen by DB_BACKEND in .env:
  - sqlite (default): one SQLite file, history.db, next to this file (local use);
  - postgres: the URL in DATABASE_URL, e.g. postgresql://user:pass@host/db?sslmode=require (Neon, Supabase).
    Needs outbound port 5432 (some networks, and Hugging Face Spaces, block it). For a hosted app
    whose disk is wiped on restart, so accounts and chats survive. The DIRECT (non-pooled) connection string
    is used: DATABASE_URL_UNPOOLED if set (Neon writes it), else DATABASE_URL. Chainlit's asyncpg driver uses
    prepared statements, which a transaction pooler (PgBouncer) breaks.

Chainlit's SQLAlchemyDataLayer writes users, threads (= chats), steps (= messages),
elements (= files) and feedbacks into these 5 tables. It doesn't create them itself,
so create_tables() does that once per process. The column names must match what
Chainlit writes, which is why they're in camelCase with quotes. Chainlit sends "tags" as a Python list,
so on Postgres that column is TEXT[]; its other JSON fields arrive as text.

Check the database afterwards:  python -X utf8 db.py

Login accounts (table "accounts", salted scrypt hashes):
    python db.py adduser NAME     create an account / reset its password (asks for the password)
    python db.py deluser NAME     remove an account (its chats are kept)
    python db.py users            list accounts
    python db.py migrate          copy everything from history.db into the Postgres in DATABASE_URL
"""

import os
import sqlite3
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

load_dotenv(Path(__file__).resolve().parent / ".env")

DB_FILE = Path(__file__).resolve().parent / "history.db"
# Postgres only on explicit opt-in (DB_BACKEND=postgres): tools such as `neon init` write DATABASE_URL into
# .env by themselves, and a network that blocks port 5432 would then make the app hang at startup.
# Neon writes both a pooled DATABASE_URL and a direct DATABASE_URL_UNPOOLED; the direct one is preferred.
_RAW_URL = (os.getenv("DATABASE_URL_UNPOOLED") or os.getenv("DATABASE_URL") or "").strip()
IS_POSTGRES = os.getenv("DB_BACKEND", "sqlite").strip().lower() == "postgres" and \
    _RAW_URL.split("://", 1)[0].split("+", 1)[0] in ("postgres", "postgresql")


# One Postgres URL -> (sync URL for psycopg, async URL for Chainlit's asyncpg, SSL required?).
# asyncpg rejects libpq options such as sslmode in the URL, so SSL is passed to Chainlit separately.
def _postgres_urls(url: str) -> tuple[str, str, bool]:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    ssl = query.get("sslmode", "") in ("require", "verify-ca", "verify-full")
    sync = urlunsplit(parts._replace(scheme="postgresql+psycopg"))
    async_query = {k: v for k, v in query.items() if k not in ("sslmode", "channel_binding")}
    async_ = urlunsplit(parts._replace(scheme="postgresql+asyncpg", query=urlencode(async_query)))
    return sync, async_, ssl


if IS_POSTGRES:
    SYNC_URL, DATABASE_URL, SSL_REQUIRE = _postgres_urls(_RAW_URL)
else:
    SYNC_URL = f"sqlite:///{DB_FILE.as_posix()}"
    DATABASE_URL = f"sqlite+aiosqlite:///{DB_FILE.as_posix()}"   # used by Chainlit (async)
    SSL_REQUIRE = False

_engine = None
_tables_ready = False


# The sync engine for our own queries (accounts, prefs, the CLI); created once.
def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(SYNC_URL, pool_pre_ping=True)
    return _engine


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
    "tags" {TAGS},
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
    "tags" {TAGS},
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


# The CREATE TABLE statements for this backend (comments dropped, one statement each).
def _schema_statements() -> list[str]:
    sql = SCHEMA.replace("{TAGS}", "TEXT[]" if IS_POSTGRES else "TEXT")
    sql = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in sql.split(";") if s.strip()]


# Create the tables if missing; runs once per process (on Postgres every call would be a network round trip).
def create_tables() -> None:
    global _tables_ready
    if _tables_ready:
        return
    with engine().begin() as conn:
        for statement in _schema_statements():
            conn.execute(text(statement))
    _tables_ready = True


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
    with engine().begin() as conn:
        conn.execute(text('INSERT INTO accounts ("username", "password_hash", "createdAt") VALUES (:u, :h, :c) '
                          'ON CONFLICT ("username") DO UPDATE SET "password_hash" = excluded."password_hash"'),
                     {"u": name, "h": hash_password(password), "c": datetime.now(timezone.utc).isoformat()})
    return name


# ---------------------------------------------------------------- per-user GUI settings
def get_pref(username: str, key: str, default: str | None = None) -> str | None:
    create_tables()
    with engine().connect() as conn:
        row = conn.execute(text('SELECT "value" FROM prefs WHERE "username" = :u AND "key" = :k'),
                           {"u": normalize_username(username), "k": key}).fetchone()
    return row[0] if row else default


def set_pref(username: str, key: str, value: str) -> None:
    create_tables()
    with engine().begin() as conn:
        conn.execute(text('INSERT INTO prefs ("username", "key", "value") VALUES (:u, :k, :v) '
                          'ON CONFLICT ("username", "key") DO UPDATE SET "value" = excluded."value"'),
                     {"u": normalize_username(username), "k": key, "v": value})


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
    with engine().begin() as conn:
        if conn.execute(text('SELECT 1 FROM users WHERE "identifier" = :u'), {"u": name}).fetchone() and \
                not conn.execute(text('SELECT 1 FROM accounts WHERE "username" = :u'), {"u": name}).fetchone():
            raise ValueError("That username is reserved. Please choose another one.")
        # DO NOTHING on a duplicate, so two simultaneous sign-ups can't overwrite each other's password.
        created = conn.execute(
            text('INSERT INTO accounts ("username", "password_hash", "createdAt") VALUES (:u, :h, :c) '
                 'ON CONFLICT ("username") DO NOTHING'),
            {"u": name, "h": hash_password(password), "c": datetime.now(timezone.utc).isoformat()}).rowcount > 0
    if not created:
        raise ValueError("That username is already taken.")
    return name


# ---- Self sign-up with a limit (the /signup page). An account made there gets the pref "signed_up" (its date,
# in the prefs table, so the schema does not change); accounts made by an admin (adduser / set_account) do not,
# so they never count. A deleted account frees its place (only signed_up rows that still have an account count).
SIGNUP_PREF = "signed_up"
_signup_lock = __import__("threading").Lock()   # the app is one process: check + create can't interleave


def signup_count() -> int:
    create_tables()
    with engine().connect() as conn:
        return conn.execute(text('SELECT COUNT(*) FROM prefs p JOIN accounts a ON a."username" = p."username" '
                                 'WHERE p."key" = :k'), {"k": SIGNUP_PREF}).scalar_one()


# Create an account from the sign-up page; limit = most self-made accounts (None = no limit).
# Raises ValueError (shown on the page) when the places are taken, or for the usual username/password rules.
def create_signup_account(username: str, password: str, limit: int | None = None) -> str:
    from datetime import datetime, timezone
    with _signup_lock:
        if limit is not None and signup_count() >= limit:
            raise ValueError("Sign-up is full: all places are taken.")
        name = create_account(username, password)
        set_pref(name, SIGNUP_PREF, datetime.now(timezone.utc).isoformat())
    return name


# Remove the login only; the user's chats stay in the database (re-adding the name gives them back).
def delete_account(username: str) -> bool:
    create_tables()
    with engine().begin() as conn:
        return conn.execute(text('DELETE FROM accounts WHERE "username" = :u'),
                            {"u": normalize_username(username)}).rowcount > 0


def list_accounts() -> list[tuple[str, str]]:
    create_tables()
    with engine().connect() as conn:
        return [tuple(r) for r in conn.execute(
            text('SELECT "username", "createdAt" FROM accounts ORDER BY "username"')).fetchall()]


# The normalized username if the password matches its account, else None.
def check_login(username: str, password: str) -> str | None:
    name = normalize_username(username)
    if not name or not password:
        return None
    create_tables()
    with engine().connect() as conn:
        row = conn.execute(text('SELECT "password_hash" FROM accounts WHERE "username" = :u'),
                           {"u": name}).fetchone()
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

        # Chainlit 2.12 stamps new chats with LOCAL time marked as UTC ('Z'); on a machine ahead of UTC a chat
        # started late in the day is then dated tomorrow and the left pane files it under "Previous 7 days".
        # Real UTC time instead, as the browser expects.
        async def get_current_timestamp(self) -> str:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"

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
    return CleanupDataLayer(conninfo=DATABASE_URL, ssl_require=SSL_REQUIRE)


# ---------------------------------------------------------------- copy history.db -> Postgres
# Tables in foreign-key order. SQLite keeps booleans as 0/1 and "tags" as JSON text; Postgres wants bool / TEXT[].
MIGRATE_TABLES = ["users", "threads", "steps", "elements", "feedbacks", "accounts", "prefs"]
BOOL_COLUMNS = {"streaming", "waitForAnswer", "isError", "defaultOpen", "autoCollapse"}


def _to_postgres(column: str, value):
    import json
    if value is None:
        return None
    if column in BOOL_COLUMNS:
        return bool(value)
    if column == "tags":
        return json.loads(value) if isinstance(value, str) else list(value)
    return value


# Copy every row of history.db into the Postgres in DATABASE_URL; rows already there (same key) are kept.
def migrate_sqlite_to_postgres() -> None:
    if not IS_POSTGRES:
        raise SystemExit("Set DATABASE_URL to a Postgres URL in .env first.")
    if not DB_FILE.exists():
        raise SystemExit(f"{DB_FILE} not found: nothing to copy.")
    create_tables()
    src = sqlite3.connect(DB_FILE)
    src.row_factory = sqlite3.Row
    with engine().begin() as conn:
        for table in MIGRATE_TABLES:
            rows = [dict(r) for r in src.execute(f'SELECT * FROM "{table}"')]
            if not rows:
                print(f"{table:<10} 0 rows")
                continue
            columns = list(rows[0].keys())
            quoted = ", ".join('"' + c + '"' for c in columns)
            values = ", ".join(f":p{i}" for i in range(len(columns)))
            sql = f'INSERT INTO "{table}" ({quoted}) VALUES ({values}) ON CONFLICT DO NOTHING'
            added = 0
            for row in rows:
                params = {f"p{i}": _to_postgres(c, row[c]) for i, c in enumerate(columns)}
                added += conn.execute(text(sql), params).rowcount
            print(f"{table:<10} {added} of {len(rows)} rows copied")
    src.close()


# Row counts per table and the latest chats, for either backend.
def show_summary() -> None:
    create_tables()
    print(f"Database: {'Postgres' if IS_POSTGRES else DB_FILE}")
    with engine().connect() as conn:
        for table in sorted(inspect(conn).get_table_names()):
            count = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            print(f"{table:<10} {count} rows")
        print("\nlatest chats:")
        for name, user, created in conn.execute(text(
                'SELECT "name", "userIdentifier", "createdAt" FROM threads ORDER BY "createdAt" DESC LIMIT 5')):
            print(f"  {created}  {user or '-':<10} {name}")


def _account_cli(args: list[str]) -> None:
    from getpass import getpass
    cmd = args[0]
    if cmd == "migrate":
        migrate_sqlite_to_postgres()
    elif cmd == "users":
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
        sys.exit("Usage: python db.py [users | adduser NAME | deluser NAME | migrate]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        _account_cli(sys.argv[1:])
        sys.exit()
    show_summary()
