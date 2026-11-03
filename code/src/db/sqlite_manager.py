import sqlite3
import os
from datetime import datetime
from typing import Optional

import config


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS institutes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                short_code  TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                om_listing_url TEXT NOT NULL,
                added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS synced_documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                institute_code  TEXT NOT NULL,
                filename        TEXT UNIQUE NOT NULL,
                pdf_url         TEXT,
                issue_date      TEXT,
                ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                superseded      INTEGER DEFAULT 0,
                superseded_by   TEXT
            );

            CREATE TABLE IF NOT EXISTS app_settings (
                key     TEXT PRIMARY KEY,
                value   TEXT NOT NULL
            );
        """)


# ──────────────────────────────────────────────
# Institute CRUD
# ──────────────────────────────────────────────

def add_institute(short_code: str, display_name: str, om_listing_url: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO institutes (short_code, display_name, om_listing_url) VALUES (?, ?, ?)",
            (short_code.lower(), display_name, om_listing_url)
        )


def update_institute(short_code: str, display_name: str, om_listing_url: str):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE institutes SET display_name=?, om_listing_url=? WHERE short_code=?",
            (display_name, om_listing_url, short_code.lower())
        )


def remove_institute(short_code: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM institutes WHERE short_code=?", (short_code.lower(),))


def list_institutes() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM institutes ORDER BY display_name").fetchall()
        return [dict(r) for r in rows]


def get_institute(short_code: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM institutes WHERE short_code=?", (short_code.lower(),)
        ).fetchone()
        return dict(row) if row else None


# ──────────────────────────────────────────────
# Document CRUD
# ──────────────────────────────────────────────

def is_already_ingested(filename: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM synced_documents WHERE filename=?", (filename,)
        ).fetchone()
        return row is not None


def mark_ingested(institute_code: str, filename: str, pdf_url: str, issue_date: str):
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO synced_documents
               (institute_code, filename, pdf_url, issue_date, ingested_at)
               VALUES (?, ?, ?, ?, ?)""",
            (institute_code.lower(), filename, pdf_url, issue_date,
             datetime.utcnow().isoformat())
        )


def mark_superseded(old_filename: str, new_filename: str):
    """Mark an existing document as superseded by a newer one."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE synced_documents SET superseded=1, superseded_by=? WHERE filename=?",
            (new_filename, old_filename)
        )


def find_document_by_date(issue_date: str, institute_code: str) -> Optional[dict]:
    """Look up a document by its issue_date and institute, used for supersession linking."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM synced_documents WHERE issue_date=? AND institute_code=?",
            (issue_date, institute_code.lower())
        ).fetchone()
        return dict(row) if row else None


def list_documents(institute_code: Optional[str] = None) -> list[dict]:
    with _get_conn() as conn:
        if institute_code:
            rows = conn.execute(
                "SELECT * FROM synced_documents WHERE institute_code=? ORDER BY issue_date DESC",
                (institute_code.lower(),)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM synced_documents ORDER BY issue_date DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def last_sync_time(institute_code: str) -> Optional[str]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(ingested_at) as last_sync FROM synced_documents WHERE institute_code=?",
            (institute_code.lower(),)
        ).fetchone()
        return row["last_sync"] if row else None


# ──────────────────────────────────────────────
# App Settings (key-value store)
# ──────────────────────────────────────────────

def get_setting(key: str, default: str = "") -> str:
    with _get_conn() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value))
        )


def document_count(institute_code: str) -> int:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM synced_documents WHERE institute_code=?",
            (institute_code.lower(),)
        ).fetchone()
        return row["cnt"] if row else 0
