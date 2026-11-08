"""Tests: jieba leg stays fresh — realtime hooks + incremental rebuild.

The 2026-08-30 finding: the one-shot jieba index went 37% stale (16,942
rows vs 26,763 facts) within a day because NOTHING synced it. Fix in
depth: ①store-level hooks keep facts_jieba in lockstep with facts
(insert/update/delete); ②a cron-side incremental rebuild catches drift
from external writers (distiller subprocesses, backfills, manual SQL)
and reports. Both paths share jieba_leg.jieba_tokenize.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
jieba = pytest.importorskip("jieba", reason="jieba not installed")

from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.jieba_leg import (
    build_jieba_index,
    jieba_candidates,
    sync_missing_jieba_rows,
    jieba_index_stats,
)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "j.db"))
    yield s
    s.close()


class TestRealtimeHooks:
    def test_add_fact_after_index_build_is_searchable(self, store):
        build_jieba_index(store)   # build once
        fid = store.add_fact("澜起科技港股代码06809", category="memory")
        hits = jieba_candidates(store, "澜起科技", top=5)
        assert any(f["fact_id"] == fid for f in hits), \
            "new fact must be jieba-searchable WITHOUT a rebuild"

    def test_update_fact_refreshes_tokens(self, store):
        build_jieba_index(store)
        fid = store.add_fact("旧内容关于玻璃基板", category="memory")
        store.update_fact(fid, content="新内容关于错杀池监控")
        hits = jieba_candidates(store, "错杀池监控", top=10)
        # jieba splits 错/杀池/监控 — match on the 监控 word instead
        assert any(f["fact_id"] == fid for f in hits)
        old = jieba_candidates(store, "玻璃基板", top=50)
        assert not any(f["fact_id"] == fid for f in old), \
            "old tokens must be GONE after content update"

    def test_remove_fact_deletes_row(self, store):
        build_jieba_index(store)
        fid = store.add_fact("待删除的飞书推送配置", category="memory")
        store.remove_fact(fid)
        hits = jieba_candidates(store, "飞书推送", top=50)
        assert not any(f["fact_id"] == fid for f in hits)

    def test_hooks_survive_without_prior_build(self, store):
        """No build_jieba_index call — hooks must not raise and must not
        create a half-table (the leg stays absent until someone builds)."""
        store.add_fact("任何内容", category="memory")  # no exception
        assert jieba_candidates(store, "任何", top=5) == []


class TestIncrementalRebuild:
    def test_sync_catches_drift(self, store):
        build_jieba_index(store)
        store.add_fact("钩子覆盖的正常行", category="memory")
        # simulate an EXTERNAL writer bypassing hooks (raw SQL)
        cur = store._conn.execute(
            "INSERT INTO facts (content, content_n, category) VALUES (?, ?, ?)",
            ("外部写入的漏网行", "外部写入的漏网行", "memory"))
        drift_id = cur.lastrowid
        store._conn.commit()

        stats = jieba_index_stats(store)
        assert stats["facts"] == 2
        assert stats["jieba_rows"] == 1, "drift visible before sync"
        n = sync_missing_jieba_rows(store)
        assert n == 1
        stats2 = jieba_index_stats(store)
        assert stats2["jieba_rows"] == stats2["facts"]
        hits = jieba_candidates(store, "漏网", top=5)
        assert any(f["fact_id"] == drift_id for f in hits)

    def test_sync_idempotent(self, store):
        build_jieba_index(store)
        store.add_fact("幂等性测试行", category="memory")
        assert sync_missing_jieba_rows(store) == 0  # hooks already did it
        assert sync_missing_jieba_rows(store) == 0

    def test_sync_on_store_without_jieba_table(self, store):
        # leg never built → sync is a no-op returning 0, never raises
        store.add_fact("x行", category="memory")
        assert sync_missing_jieba_rows(store) == 0

    def test_stats_on_missing_table(self, store):
        stats = jieba_index_stats(store)
        assert stats == {"facts": 0, "jieba_rows": 0, "drift": 0}
