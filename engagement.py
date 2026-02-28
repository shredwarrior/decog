"""
engagement.py  –  Engagement, Sharing & Storage
================================================
Flask Blueprint that provides:
  - SQLite storage for every analysis
  - Email subscription (Cognition Today branded)
  - Minimal share page with Open Graph meta tags
"""

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

from flask import (
    Blueprint, request, jsonify, session,
    render_template, current_app, make_response
)

engagement = Blueprint("engagement", __name__)

DB_PATH = os.getenv("DB_PATH", "argument_analyser.db")


# ──────────────────────────────────────────────
#  Database helpers
# ──────────────────────────────────────────────

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Call once on app startup."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id              TEXT PRIMARY KEY,
            argument_text   TEXT NOT NULL,
            score           INTEGER,
            raw_score       REAL,
            razor_alignment REAL,
            status_label    TEXT,
            status_message  TEXT,
            executive_summary TEXT,
            detected_issues TEXT,
            metadata        TEXT,
            created_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS subscribers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT UNIQUE NOT NULL,
            created_at  TEXT NOT NULL
        );
    """)
    # Migrate: add columns if missing (safe for existing DBs)
    cursor = conn.execute("PRAGMA table_info(analyses)")
    existing = {row[1] for row in cursor.fetchall()}
    if "score_breakdown" not in existing:
        conn.execute("ALTER TABLE analyses ADD COLUMN score_breakdown TEXT DEFAULT '{}'")
    if "improvements" not in existing:
        conn.execute("ALTER TABLE analyses ADD COLUMN improvements TEXT DEFAULT '[]'")
    conn.commit()
    conn.close()


def save_analysis(argument_text, result):
    """
    Persist a completed analysis and return its UUID.
    Called from the /api/analyze route after a successful run.
    """
    analysis_id = str(uuid.uuid4())
    breakdown = result.get("score_breakdown", {})
    exec_sentence = ""
    if result.get("detected_issues"):
        exec_sentence = result["detected_issues"].get("executive_summary_sentence", "")

    conn = _get_db()
    conn.execute(
        """INSERT INTO analyses
           (id, argument_text, score, raw_score, razor_alignment,
            status_label, status_message, executive_summary,
            detected_issues, metadata, score_breakdown, improvements,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            analysis_id,
            argument_text,
            result.get("score", 0),
            breakdown.get("raw_score", 0),
            breakdown.get("razor_alignment", 0),
            breakdown.get("status_label", ""),
            breakdown.get("status_message", ""),
            exec_sentence,
            json.dumps(result.get("detected_issues", {})),
            json.dumps(result.get("metadata", [])),
            json.dumps(breakdown),
            json.dumps(result.get("improvements", [])),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return analysis_id


def _get_analysis(analysis_id):
    conn = _get_db()
    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

@engagement.route("/api/subscribe", methods=["POST"])
def subscribe():
    """Accept an email, store it, flag the session as subscribed."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"success": False, "error": "Please enter a valid email address."}), 400

    conn = _get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO subscribers (email, created_at) VALUES (?, ?)",
            (email, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()

    session.permanent = True
    session["subscribed"] = True
    session["subscriber_email"] = email

    return jsonify({
        "success": True,
        "message": "Welcome to Cognition Today! You now have access to 2,000-word analysis.",
        "max_words": 2000,
    })


def _no_session_response(html):
    """Wrap rendered HTML in a response that deletes the session cookie."""
    resp = make_response(html)
    resp.delete_cookie(
        current_app.config.get("SESSION_COOKIE_NAME", "session"),
        path="/",
    )
    return resp


@engagement.route("/api/share/<analysis_id>")
def share_page(analysis_id):
    """Minimal public page for social media link previews."""
    row = _get_analysis(analysis_id)
    if not row:
        return "Analysis not found.", 404

    breakdown = json.loads(row.get("score_breakdown", "{}"))

    html = render_template(
        "share.html",
        analysis=row,
        breakdown=breakdown,
        detected=json.loads(row.get("detected_issues", "{}")),
    )
    return _no_session_response(html)


@engagement.route("/analysis/<analysis_id>")
def public_analysis(analysis_id):
    """Full public view of a shared analysis."""
    row = _get_analysis(analysis_id)
    if not row:
        return "Analysis not found.", 404

    detected = json.loads(row.get("detected_issues", "{}"))
    breakdown = json.loads(row.get("score_breakdown", "{}"))
    improvements = json.loads(row.get("improvements", "[]"))

    words = (row.get("argument_text") or "").split()
    truncated_arg = " ".join(words[:300]) + ("..." if len(words) > 300 else "")

    html = render_template(
        "public_analysis.html",
        analysis=row,
        detected=detected,
        breakdown=breakdown,
        improvements=improvements,
        truncated_arg=truncated_arg,
    )
    return _no_session_response(html)


