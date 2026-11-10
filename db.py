"""The database that stores chat history (the left panel), as one SQLite file: history.db.

Chainlit's SQLAlchemyDataLayer writes users, threads (= chats), steps (= messages),
elements (= files) and feedbacks into these 5 tables. It doesn't create them itself,
so create_tables() does that once on startup. The column names must match what
Chainlit writes, which is why they're in camelCase with quotes.

SQLite is a single file, which is fine locally. For a hosted app whose disk gets
wiped on restart, switch DATABASE_URL to a free hosted Postgres (e.g. Neon/Supabase).

Check the file afterwards:  .venv/Scripts/python.exe -X utf8 db.py
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
"""


def create_tables() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    create_tables()
    with sqlite3.connect(DB_FILE) as conn:
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            print(f"{table:<10} {count} rows")
        print("\nlatest chats:")
        for name, user, created in conn.execute(
                'SELECT "name", "userIdentifier", "createdAt" FROM threads ORDER BY "createdAt" DESC LIMIT 5'):
            print(f"  {created}  {user:<10} {name}")
