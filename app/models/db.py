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
        conn.executescript(SCHEMA)
