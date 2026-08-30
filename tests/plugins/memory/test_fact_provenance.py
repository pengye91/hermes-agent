"""TDD tests for fact provenance (msg_id) — the 2026-08-29 double-distill
incident's design blind spot: facts table records no msg_id, so the
+1,161 redundant facts from a concurrent run could never be precisely
excised, and 'which message produced this fact' was unanswerable.

Contract:
1. facts.msg_id column exists (auto-migrated on open for legacy stores).
2. add_fact(..., msg_id=N) persists N; default None stays None.
3. Duplicate content returns the EXISTING fact_id and (if the new call
   carries msg_id and the old row lacks it) backfills msg_id — provenance
   recovered on re-import, never overwritten with NULL.
4. Distillation pipeline passes msg_id through (distill_llm side is
   covered by its own suite; here we pin the store contract).
"""

import sys

HERMES = "/usr/local/lib/hermes-agent"
if HERMES not in sys.path:
    sys.path.insert(0, HERMES)

from plugins.memory.holographic.store import MemoryStore


def _cols(store):
    return {r[1] for r in store._conn.execute("PRAGMA table_info(facts)")}


def test_msg_id_column_exists(tmp_path):
    store = MemoryStore(db_path=tmp_path / "t.db")
    assert "msg_id" in _cols(store)
    store.close()


def test_add_fact_persists_msg_id(tmp_path):
    store = MemoryStore(db_path=tmp_path / "t.db")
    fid = store.add_fact("智谱是2513.HK", category="corpus", msg_id=46485)
    row = store._conn.execute(
        "SELECT msg_id FROM facts WHERE fact_id=?", (fid,)).fetchone()
    assert row[0] == 46485
    store.close()


def test_add_fact_default_msg_id_none(tmp_path):
    store = MemoryStore(db_path=tmp_path / "t.db")
    fid = store.add_fact("无来源事实", category="general")
    row = store._conn.execute(
        "SELECT msg_id FROM facts WHERE fact_id=?", (fid,)).fetchone()
    assert row[0] is None
    store.close()


def test_duplicate_backfills_missing_provenance(tmp_path):
    store = MemoryStore(db_path=tmp_path / "t.db")
    fid1 = store.add_fact("重复内容A", category="corpus")            # no msg_id
    fid2 = store.add_fact("重复内容A", category="corpus", msg_id=99)  # same content
    assert fid2 == fid1  # dedup returns existing
    row = store._conn.execute(
        "SELECT msg_id FROM facts WHERE fact_id=?", (fid1,)).fetchone()
    assert row[0] == 99  # backfilled


def test_duplicate_does_not_overwrite_existing_provenance(tmp_path):
    store = MemoryStore(db_path=tmp_path / "t.db")
    fid1 = store.add_fact("重复内容B", category="corpus", msg_id=7)
    fid2 = store.add_fact("重复内容B", category="corpus", msg_id=8)
    assert fid2 == fid1
    row = store._conn.execute(
        "SELECT msg_id FROM facts WHERE fact_id=?", (fid1,)).fetchone()
    assert row[0] == 7  # first provenance wins


def test_legacy_store_migrates_msg_id(tmp_path):
    # create a legacy store WITHOUT msg_id, then reopen with new code
    import sqlite3
    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(p)
    conn.executescript("""
        CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT UNIQUE,
            category TEXT DEFAULT 'general', tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5, retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')));
        INSERT INTO facts(content) VALUES ('老事实');
    """)
    conn.commit()
    conn.close()
    store = MemoryStore(db_path=p)  # migration on open
    assert "msg_id" in _cols(store)
    n = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert n == 1  # data preserved
    store.close()
