"""Tests for NFKC normalization + code-token protection in the FTS query
sanitizer (cycle A).

Evidence (2026-08-30 forensics): full-width chars pass through untouched
('Ａ股暴跌诊断' → shingles lose the Ａ entirely; 'ＧＢＭ' never matches
'GBM' corpus) and case is never folded ('minimax' must rely on trigram
coincidence to hit 'MiniMax'). Chinese tech chat is FULL of full-width
letters/punctuation (WeChat/Feishu copy-paste). CLS production solved
this exact class years ago; the holo side never got it.

NOTE: _sanitize_fts_query returns an FTS5 OR EXPRESSION STRING
('"shingle" OR "token"'), not a list — tests parse quoted literals.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.retrieval import FactRetriever


def quoted(query: str) -> list[str]:
    """Extract quoted phrase literals from the OR expression."""
    out = FactRetriever._sanitize_fts_query(query)
    return re.findall(r'"([^"]+)"', out)


class TestNfkcNormalization:
    def test_fullwidth_ascii_folded(self):
        toks = quoted("Ａ股暴跌诊断")
        # NFKC turns Ａ股 into 'a'+股 glued: the CJK shingles must survive
        # (they carry the match); the lone 'a' is stopword-class and is
        # correctly dropped. End-to-end matching is covered by
        # test_migrated_legacy_store_backfills_content_n.
        assert any("股暴跌" == t or "暴跌诊" == t for t in toks), toks

    def test_fullwidth_latin_word_matches_ascii_corpus(self):
        toks = quoted("ＧＢＭ模型")
        assert any("gbm" in t for t in toks), toks

    def test_case_fold_mixed_brand(self):
        toks = quoted("minimax的港股代码")
        assert any("minimax" in t for t in toks), toks

    def test_ascii_punctuation_fullwidth_folded(self):
        toks = quoted("错杀池：数据库？")
        assert toks and all("：" not in t and "？" not in t for t in toks), toks

    def test_pure_cjk_unchanged_by_nfkc(self):
        toks = quoted("百合花纯度门槛")
        assert any("百合花" in t for t in toks), toks


class TestCodeTokenProtection:
    def test_stock_code_kept_whole(self):
        toks = quoted("智谱 2513.HK 港股代码")
        assert any("2513.hk" in t for t in toks), toks

    def test_pure_numeric_code_kept_whole(self):
        toks = quoted("茂莱光学 688502")
        assert any("688502" in t for t in toks), toks

    def test_version_number_kept_whole(self):
        toks = quoted("v1.9.0 止盈规则")
        assert any("v1.9.0" in t for t in toks), toks

    def test_short_code_not_fragmented_by_trigram(self):
        toks = quoted("06809.HK")
        assert any("06809.hk" in t for t in toks), toks

    def test_cjk_shingles_still_emitted_alongside(self):
        toks = quoted("澜起科技 06809.HK")
        assert any(all("\u4e00" <= c <= "\u9fff" for c in t) for t in toks), toks


class TestIndexSideNormalization:
    def test_fullwidth_corpus_matches_ascii_query(self, tmp_path):
        """End-to-end: query 'GBM' must retrieve a fact written as 'ＧＢＭ'.
        Requires the INDEX side (content_n) to normalize too."""
        from plugins.memory.holographic.store import MemoryStore
        store = MemoryStore(str(tmp_path / "nfkc.db"))
        store.add_fact("ＧＢＭ模型遇到429时切换到minimax", category="tool")
        ret = FactRetriever(store=store)
        hits = ret._fts_candidates("GBM模型 429", None, 0.0, 5)
        assert any("429" in f["content"] for f in hits), \
            [f["content"][:50] for f in hits]
        store.close()

    def test_migrated_legacy_store_backfills_content_n(self, tmp_path):
        """A store created pre-content_n (simulate by dropping the column's
        data) must backfill on open — migration path for the 17k prod DB."""
        import sqlite3
        from plugins.memory.holographic.store import MemoryStore, normalize_for_fts
        db = tmp_path / "legacy.db"
        store = MemoryStore(str(db))
        store.add_fact("Ａ股暴跌诊断skill入口", category="memory")
        store.close()
        # simulate legacy: wipe content_n + rebuild FTS without it
        conn = sqlite3.connect(db)
        conn.execute("UPDATE facts SET content_n = NULL")
        conn.execute("DROP TABLE facts_fts")
        conn.execute("""CREATE VIRTUAL TABLE facts_fts
                        USING fts5(content, tags, tokenize='trigram')""")
        conn.execute("""INSERT INTO facts_fts(rowid, content, tags)
                        SELECT fact_id, content, tags FROM facts""")
        conn.commit(); conn.close()
        # reopen → migration should backfill + rebuild with content_n
        store2 = MemoryStore(str(db))
        row = store2._conn.execute(
            "SELECT content_n FROM facts LIMIT 1").fetchone()
        assert row and normalize_for_fts("Ａ股暴跌诊断skill入口") == row[0]
        ret = FactRetriever(store=store2)
        hits = ret._fts_candidates("A股暴跌", None, 0.0, 5)
        assert any("暴跌" in f["content"] for f in hits)
        store2.close()
