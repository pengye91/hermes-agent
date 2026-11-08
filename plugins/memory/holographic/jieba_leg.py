"""Jieba word-level FTS leg — additive precision leg (cycle 6 + sync hardening).

The trigram FTS leg nails recall but its precision is structurally weak
(2-char subjects cross-match inside longer words). This leg adds a
word-level FTS5 table whose rows are jieba.cut_for_search tokens —
index side and query side share the same tokenizer.

2026-08-30 sync hardening: the first implementation used a plain table
+ external-content FTS + FULL REBUILD per row — 128ms/add at 1k rows and
O(n) worsening. Rewritten as ONE contentful FTS5 table with
rowid=fact_id: row-level INSERT/DELETE, zero rebuilds, ~1ms/add. Kept
fresh by ①store hooks (add/update/remove) and ②a daily incremental sync
(jieba_sync_daily.sh) that catches external writers.

Additive by design: the table is built explicitly (``build_jieba_index``),
queried explicitly (``jieba_candidates``). Stores without the table
return [] — never raise — so existing deployments are untouched.
"""
from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

_TABLE = "facts_jieba_fts"

_jieba = None
_ready = False


def _get_jieba():
    global _jieba, _ready
    if _ready:
        return _jieba
    _ready = True
    try:
        import jieba
        jieba.setLogLevel(60)  # silence the loading banner
        _jieba = jieba
    except ImportError:
        _jieba = None
    return _jieba


def jieba_tokenize(text) -> list[str]:
    """Search-mode tokens (full words + sub-words), non-CJK runs kept."""
    jb = _get_jieba()
    if jb is None or not text:
        return []
    return [t for t in jb.cut_for_search(text) if t.strip()]


def _table_exists(conn) -> bool:
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (_TABLE,)).fetchone() is not None


def _leg_is_contentful(conn) -> bool:
    """True when facts_jieba_fts is OUR shape (contentful, tokens stored
    in-row). False for the legacy external-content shape (content=
    'facts_jieba') — which silently swallows direct INSERTs into shadow
    tables while MATCH reads the stale external table (2026-08-31 prod
    incident: sync claimed 10,546 rows, drift stayed 10,546)."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name=?", (_TABLE,)).fetchone()
    if row is None:
        return False
    ddl = (row[0] or "").lower()
    return "content=" not in ddl


def _ensure_leg_shape(store) -> bool:
    """Ensure the leg is the contentful shape; rebuild legacy shapes.
    Returns True when the (new) table exists afterwards."""
    with store._lock:
        conn = store._conn
        if _table_exists(conn) and _leg_is_contentful(conn):
            return True
        # legacy external-content shape (or plain facts_jieba remnant) —
        # drop both, caller must rebuild via build_jieba_index
        if _table_exists(conn) or _legacy_plain_exists(conn):
            conn.execute("DROP TABLE IF EXISTS facts_jieba_fts")
            conn.execute("DROP TABLE IF EXISTS facts_jieba")
            conn.commit()
    return False


def _legacy_plain_exists(conn) -> bool:
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='facts_jieba'"
    ).fetchone() is not None


def build_jieba_index(store, batch: int = 500) -> int:
    """(Re)build the jieba FTS table over all facts. Returns rows indexed.

    Contentful FTS5 with rowid=fact_id: row-level ops everywhere, no
    external-content rebuild dance. Drops the legacy two-table shape
    (facts_jieba + external fts) if present.
    """
    jb = _get_jieba()
    if jb is None:
        return 0
    with store._lock:
        conn = store._conn
        # legacy shape cleanup (2026-08-30 first implementation)
        conn.execute("DROP TABLE IF EXISTS facts_jieba")
        conn.execute(f"DROP TABLE IF EXISTS {_TABLE}")
        conn.execute(
            f"CREATE VIRTUAL TABLE {_TABLE} USING fts5(tokens, tokenize='unicode61')")
        rows = conn.execute("SELECT fact_id, content FROM facts").fetchall()
        for i in range(0, len(rows), batch):
            conn.executemany(
                f"INSERT INTO {_TABLE}(rowid, tokens) VALUES (?, ?)",
                [(fid, " ".join(jieba_tokenize(content or "")))
                 for fid, content in rows[i:i + batch]])
        conn.commit()
    return len(rows)


def jieba_upsert_row(store, fact_id: int, content: str) -> None:
    """Insert-or-replace ONE fact's jieba tokens (realtime hook, O(row)).

    No-op when the leg was never built — the leg must not spring into
    existence from a hook (build_jieba_index is the explicit opt-in).
    """
    jb = _get_jieba()
    if jb is None:
        return
    with store._lock:
        if not _table_exists(store._conn):
            return
        store._conn.execute(f"DELETE FROM {_TABLE} WHERE rowid = ?", (fact_id,))
        store._conn.execute(
            f"INSERT INTO {_TABLE}(rowid, tokens) VALUES (?, ?)",
            (fact_id, " ".join(jieba_tokenize(content or ""))))
        store._conn.commit()


def jieba_delete_row(store, fact_id: int) -> None:
    """Remove ONE fact's jieba row (realtime hook; no-op without the leg)."""
    jb = _get_jieba()
    if jb is None:
        return
    with store._lock:
        if not _table_exists(store._conn):
            return
        store._conn.execute(f"DELETE FROM {_TABLE} WHERE rowid = ?", (fact_id,))
        store._conn.commit()


def jieba_index_stats(store) -> dict:
    """{facts, jieba_rows, drift} — drift = facts missing from the leg."""
    with store._lock:
        facts = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        if not _table_exists(store._conn):
            return {"facts": facts, "jieba_rows": 0, "drift": 0}
        rows = store._conn.execute(
            f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
    return {"facts": facts, "jieba_rows": rows, "drift": max(0, facts - rows)}


def sync_missing_jieba_rows(store, batch: int = 500) -> int:
    """Incremental sync: index facts absent from the jieba leg (external
    writers, pre-hook data). Returns rows added.

    Auto-migrates legacy external-content shapes by REBUILDING the whole
    leg (they silently swallow INSERTs — see _leg_is_contentful); in that
    case the return equals the full fact count.
    """
    jb = _get_jieba()
    if jb is None:
        return 0
    # no leg of ANY shape → the store never opted in; sync must not
    # spring the leg into existence (build_jieba_index is the opt-in)
    with store._lock:
        ours = _table_exists(store._conn)
        legacy = _legacy_plain_exists(store._conn)
    if not ours and not legacy:
        return 0
    if _leg_is_ours(store):
        return _incremental_sync(store, batch)
    # legacy shape present → rebuild as the migration path
    return build_jieba_index(store, batch)


def _leg_is_ours(store) -> bool:
    """The contentful facts_jieba_fts exists (our shape)."""
    with store._lock:
        return _table_exists(store._conn) and _leg_is_contentful(store._conn)


def _incremental_sync(store, batch: int) -> int:
    with store._lock:
        rows = store._conn.execute(
            f"SELECT f.fact_id, f.content FROM facts f "
            f"LEFT JOIN {_TABLE} j ON j.rowid = f.fact_id "
            f"WHERE j.rowid IS NULL").fetchall()
        if not rows:
            return 0
        for i in range(0, len(rows), batch):
            store._conn.executemany(
                f"INSERT OR REPLACE INTO {_TABLE}(rowid, tokens) VALUES (?, ?)",
                [(fid, " ".join(jieba_tokenize(c or "")))
                 for fid, c in rows[i:i + batch]])
        store._conn.commit()
    return len(rows)


def jieba_candidates(store, query: str, top: int = 10) -> list[dict]:
    """Query the jieba leg. [] when table/tokenizer/query missing."""
    jb = _get_jieba()
    if jb is None or not query:
        return []
    tokens = jieba_tokenize(query)
    if not tokens:
        return []
    try:
        with store._lock:
            if not _table_exists(store._conn):
                return []
            # OR-join token literals (same recall shape as the trigram leg);
            # ranking stays FTS bm25. rowid IS fact_id by construction.
            literal = " OR ".join(f'"{t}"' for t in tokens if t)
            if not literal:
                return []
            rows = store._conn.execute(
                f"SELECT ft.rowid AS fid, f.content, f.category, f.trust_score, "
                f"bm25({_TABLE}) AS rank "
                f"FROM {_TABLE} ft JOIN facts f ON f.fact_id = ft.rowid "
                f"WHERE {_TABLE} MATCH ? ORDER BY rank LIMIT ?",
                (literal, top)).fetchall()
        return [
            {"fact_id": r[0], "content": r[1], "category": r[2],
             "trust_score": r[3], "fts_rank": -r[4]}
            for r in rows
        ]
    except sqlite3.Error as e:
        logger.debug("jieba leg query failed: %s", e)
        return []
