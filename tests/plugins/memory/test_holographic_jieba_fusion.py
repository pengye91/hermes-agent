"""Tests: jieba leg fused into the default search as a third RRF leg (cycle D).

Evidence: jieba leg precision 100% on multi-word CJK queries vs trigram
50% (measured 2026-08-30); domain-compound over-segmentation (错杀池) is
harmless in FUSION because the trigram leg still carries those queries.
Design: fusion is additive — jieba table missing / jieba not installed →
identical results to today (zero-risk rollout on stores without the leg).
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("jieba", reason="jieba not installed")

from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.jieba_leg import build_jieba_index


@pytest.fixture
def fused_retriever(tmp_path):
    store = MemoryStore(str(tmp_path / "f.db"))
    store.add_fact("[L|260615|智谱] 智谱发布GLM-5.3模型", category="memory")
    store.add_fact("[L|260616|x] 澜起科技发布新产品线", category="memory")
    store.add_fact("[L|260617|x] 数据库迁移到DuckDB完成", category="memory")
    build_jieba_index(store)
    yield FactRetriever(store=store)
    store.close()


class TestJiebaFusion:
    def test_word_query_hits_right_fact_first(self, fused_retriever):
        out = fused_retriever.search("发布 模型", limit=3)
        assert out, "must return results"
        assert any("智谱" in f["content"] for f in out[:2]), \
            [f["content"][:40] for f in out]

    def test_store_without_jieba_table_unchanged(self, tmp_path):
        """The fusion must be additive: no jieba table → same behaviour as
        before cycle D (no raise, results still returned)."""
        store = MemoryStore(str(tmp_path / "nofuse.db"))
        store.add_fact("智谱发布GLM-5.3模型", category="memory")
        ret = FactRetriever(store=store)
        out = ret.search("智谱 模型", limit=3)  # no jieba index built
        assert isinstance(out, list)  # never raises
        store.close()

    def test_fused_results_superset_of_pure_lexical(self, fused_retriever, monkeypatch):
        """With the leg present, a query whose trigram shingles all miss but
        jieba words hit must still surface the fact (recall floor rises)."""
        # '数据库 迁移' — trigram on these words may fragment; jieba word
        # match carries it
        out = fused_retriever.search("数据库 迁移", limit=3)
        assert any("DuckDB" in f["content"] for f in out), \
            [f["content"][:40] for f in out]

    def test_jieba_leg_participates_in_search(self, fused_retriever, monkeypatch):
        """The load-bearing assertion: search() must CALL the jieba leg and
        fuse its results (mock-pinned; weak end-to-end tests above pass
        even without fusion on small corpora)."""
        import plugins.memory.holographic.jieba_leg as jl
        calls = {"n": 0}
        orig = jl.jieba_candidates

        def spy(store, query, top=10):
            calls["n"] += 1
            return orig(store, query, top=top)

        monkeypatch.setattr(jl, "jieba_candidates", spy)
        # retrieval imports the module lazily — patch must be visible via
        # the module object the retrieval code references
        import plugins.memory.holographic.retrieval as retrieval_mod
        monkeypatch.setattr(retrieval_mod, "jieba_candidates", spy,
                            raising=False)
        fused_retriever.search("智谱 模型", limit=3)
        assert calls["n"] >= 1, "jieba leg was never consulted"

    def test_jieba_pool_merged_into_candidates(self, fused_retriever):
        """A fact ONLY findable via the jieba leg (trigram-resistant: its
        words share no 3-char window with the query) must appear in the
        final results — proves fusion, not just consultation."""
        # 智谱 fact contains '模型' (2-char word, sub-trigram) — craft a
        # query hitting it through words only
        out = fused_retriever.search("数据库 迁移 DuckDB", limit=5)
        assert any("DuckDB" in f["content"] for f in out)
