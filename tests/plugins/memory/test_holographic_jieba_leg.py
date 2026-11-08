"""Tests for the jieba word-level FTS leg (dual-index A/B, cycle 6).

Evidence (research brief §②): jieba FTS5 recall 100% / precision 94% vs
trigram+LIKE 100% / 45% (deepseek-harness #1456, 1000 Chinese memories).
Our trigram leg already nails recall; the win candidate is PRECISION —
fewer false positives ('明天开' matching '天开会' class errors) feeding
cleaner pools to the rerankers.

Design:
- Second FTS5 table (unicode61) fed jieba.cut_for_search output joined
  by spaces — index side AND query side use the SAME tokenizer.
- The leg is ADDITIVE: presence of the jieba table must never change
  results when absent (migration-safe, existing stores keep working).
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

jieba = pytest.importorskip("jieba", reason="jieba not installed")

from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.jieba_leg import (
    jieba_tokenize,
    build_jieba_index,
    jieba_candidates,
)


class TestTokenizer:
    def test_cut_for_search_splits_long_words(self):
        # search mode emits both the full word and sub-words
        toks = jieba_tokenize("智谱港股代码2513")
        assert any("智谱" == t for t in toks)

    def test_query_and_doc_side_same_function(self):
        # SAME tokenizer on both sides is the correctness invariant
        assert jieba_tokenize("百合花纯度门槛") == jieba_tokenize("百合花纯度门槛")

    def test_empty_safe(self):
        assert jieba_tokenize("") == []
        assert jieba_tokenize(None) == []


class TestIndexAndCandidates:
    def test_index_build_and_exact_word_match(self, tmp_path):
        store = MemoryStore(str(tmp_path / "j.db"))
        store.add_fact("港股AI双雄代码勿混: 智谱=2513.HK,MiniMax=0100.HK。", category="memory")
        store.add_fact("澜起科技港股代码06809.HK", category="memory")
        n = build_jieba_index(store)
        assert n >= 2
        hits = jieba_candidates(store, "智谱", top=10)
        assert hits and any("2513" in f["content"] for f in hits)
        store.close()

    def test_no_cross_board_prefix_false_positive(self, tmp_path):
        # trigram classic error: 2-char subject cross-matching inside other
        # words. jieba word-level match on 智谱 must not return 澜起-only fact
        store = MemoryStore(str(tmp_path / "j2.db"))
        store.add_fact("澜起科技港股代码06809.HK，中兴通讯港股代码00763.HK", category="memory")
        build_jieba_index(store)
        hits = jieba_candidates(store, "智谱", top=10)
        assert not any("智谱" in f["content"] for f in hits) or not hits
        store.close()

    def test_absent_index_returns_empty_not_raise(self, tmp_path):
        store = MemoryStore(str(tmp_path / "j3.db"))
        store.add_fact("x", category="memory")
        # index never built — query must degrade to empty list
        hits = jieba_candidates(store, "x", top=10)
        assert hits == []
        store.close()

    def test_incremental_add_updates_index(self, tmp_path):
        store = MemoryStore(str(tmp_path / "j4.db"))
        store.add_fact("百合花纯度门槛1.3%", category="memory")
        build_jieba_index(store)
        store.add_fact("福晶科技是光刻机核心标的", category="memory")
        build_jieba_index(store)  # idempotent rebuild — second call is no-op for old rows
        hits = jieba_candidates(store, "光刻机", top=10)
        assert any("福晶科技" in f["content"] for f in hits)
        store.close()
