"""TDD tests for wiring retrieval_count into the fact_store search path.

2026-08-30 audit: helpful_count>0 on 0/27,493 facts, retrieval_count>0
on 10 — the learning loop (trust feedback) never sees traffic because
the MAIN search path (FactRetriever.search, used by fact_store) never
increments retrieval_count; only the legacy store.search_facts does.

Contract:
1. FactRetriever.search increments retrieval_count for every RETURNED
   fact (they were surfaced to the caller — that is a retrieval event).
2. probe/related/reason also increment (same argument).
3. repeated searches accumulate (counter += 1 per call).
4. failure of the counter UPDATE must NEVER fail the search itself
   (metrics are best-effort; wrap in try/except).
"""

import sys

HERMES = "/usr/local/lib/hermes-agent"
if HERMES not in sys.path:
    sys.path.insert(0, HERMES)

from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


def _mk(tmp_path):
    store = MemoryStore(db_path=tmp_path / "t.db", default_trust=0.6, hrr_dim=1024)
    fid = store.add_fact(content="百合花是颜料公司而非光刻胶公司", category="corpus")
    return store, fid


def test_search_increments_retrieval_count(tmp_path):
    store, fid = _mk(tmp_path)
    r = FactRetriever(store=store)
    r.search("百合花 颜料", limit=5)
    n = store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id=?", (fid,)).fetchone()[0]
    assert n >= 1
    store.close()


def test_repeated_searches_accumulate(tmp_path):
    store, fid = _mk(tmp_path)
    r = FactRetriever(store=store)
    r.search("百合花", limit=5)
    r.search("颜料 公司", limit=5)
    n = store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id=?", (fid,)).fetchone()[0]
    assert n >= 2
    store.close()


def test_probe_increments_retrieval_count(tmp_path):
    store, fid = _mk(tmp_path)
    r = FactRetriever(store=store)
    r.probe("百合花", limit=5)
    n = store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id=?", (fid,)).fetchone()[0]
    assert n >= 1
    store.close()


def test_counter_failure_never_breaks_search(tmp_path):
    """Counter UPDATE fails (table renamed) — search must still return."""
    store, fid = _mk(tmp_path)
    r = FactRetriever(store=store)
    store._conn.execute("ALTER TABLE facts RENAME TO facts_x")
    try:
        results = r.search("百合花", limit=5)  # must not raise
        assert isinstance(results, list)
    finally:
        store._conn.execute("ALTER TABLE facts_x RENAME TO facts")
    store.close()
