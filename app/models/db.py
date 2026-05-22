import sqlite3
from app.config import settings
import os
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS mentions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    text          TEXT NOT NULL,
    source        TEXT,
    published_at  TEXT,
    reach         INTEGER DEFAULT 0,
    sentiment     TEXT,
    sentiment_score REAL,
    topics        TEXT,          -- JSON array string
    summary       TEXT,
    web3_signals  TEXT,          -- JSON dict string
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sentiment ON mentions(sentiment);
CREATE INDEX IF NOT EXISTS idx_published ON mentions(published_at);
CREATE INDEX IF NOT EXISTS idx_reach ON mentions(reach);

CREATE TABLE IF NOT EXISTS slack_subscribers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    webhook_url   TEXT UNIQUE NOT NULL,
    created_at    TEXT DEFAULT (datetime('now'))
);
"""

def get_conn():
    os.makedirs(os.path.dirname(os.path.abspath(settings.SQLITE_DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_conn():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with db_conn() as conn:
        # Check if table exists
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mentions'"
        ).fetchone()
        
        if table_exists:
            # Check existing columns
            cursor = conn.execute("PRAGMA table_info(mentions)")
            columns = [row["name"] for row in cursor.fetchall()]
            
            # If any required column is missing, recreate table
            required_cols = ["id", "title", "text", "source", "published_at", "reach", "sentiment", "sentiment_score", "topics", "summary", "web3_signals"]
            missing = [c for c in required_cols if c not in columns]
            if missing:
                conn.execute("DROP TABLE mentions")
        
        conn.executescript(SCHEMA)
