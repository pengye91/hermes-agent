"""Tests for the distillation qualifier-retention audit (cycle 8).

Evidence (research brief §④): CogCanvas measured verbatim 93% vs
summarization 19% exact-match on constraint retention ('use type hints
EVERYWHERE' → 'prefers type hints' loses the quantifier). NovaCortex
fixed their equivalent with a date-preservation prompt clause + audit.

Our distiller prompt demands qualifier retention (rule 2: action verbs
不参与/减仓/清仓; rule 3: 数字/日期) but nothing VERIFIES it. This
audit samples distilled facts against their SOURCE messages (via the
msg_id provenance column) and reports retention rates per class:
- dates:  source has explicit date/percent/number → fact keeps a number
- negatives: source has 不/勿/无/未/别  → fact keeps a negation token
- quantifiers: source has 全部/只/均/每/最多/至少 → fact keeps one
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.distill_audit import (
    extract_qualifiers,
    audit_pair,
    sample_audit,
    NEG_PAT,
    DATE_PAT,
    QUANT_PAT,
)


class TestPatterns:
    def test_negation_tokens(self):
        assert NEG_PAT.search("不参与百合花")
        assert NEG_PAT.search("清仓后勿回头")
        assert NEG_PAT.search("无实质内容")
        assert not NEG_PAT.search("参与百合花定增")

    def test_dates_and_numbers(self):
        assert DATE_PAT.search("2026年6月买入")
        assert DATE_PAT.search("纯度门槛1.3%")
        assert DATE_PAT.search("2513.HK")
        assert not DATE_PAT.search("没有任何数字")

    def test_quantifiers(self):
        assert QUANT_PAT.search("全部清仓")
        assert QUANT_PAT.search("只适合小仓位")
        assert QUANT_PAT.search("每季度复核")
        assert not QUANT_PAT.search("适合小仓位")


class TestExtract:
    def test_extracts_all_classes(self):
        q = extract_qualifiers("2026年6月决定不参与，只保留观察仓，全部放弃定增")
        assert q["dates"] and q["negatives"] and q["quantifiers"]

    def test_empty_when_none(self):
        q = extract_qualifiers("今天聊了百合花")
        assert q["dates"] == [] and q["negatives"] == [] and q["quantifiers"] == []


class TestAuditPair:
    def test_perfect_retention(self):
        src = "2026年6月决定不参与百合花定增"
        fact = "百合花定增: 不参与 (2026年6月决策)"
        r = audit_pair(src, fact)
        assert r["dates_retained"] == 1.0
        assert r["negatives_retained"] == 1.0

    def test_lost_negation_is_flagged(self):
        src = "决定不参与百合花定增，全部放弃"
        fact = "百合花定增参与决策"  # negation AND quantifier dropped
        r = audit_pair(src, fact)
        assert r["negatives_retained"] == 0.0
        assert r["quantifiers_retained"] == 0.0

    def test_partial_number_retention(self):
        # class-level retention: SOME date-class tokens kept → 1.0
        src = "纯度门槛1.3%，2027年才建成"
        fact = "纯度门槛1.3%达标"
        r = audit_pair(src, fact)
        assert r["dates_retained"] == 1.0  # class-level: any member kept

    def test_no_source_qualifiers_is_full_retention(self):
        r = audit_pair("聊了天气", "天气话题")
        assert r["dates_retained"] == 1.0  # vacuous truth — nothing to lose


class TestSampleAudit:
    def test_aggregates_and_reports(self, tmp_path):
        import sqlite3
        db = tmp_path / "a.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT, msg_id INTEGER)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        # msg 1: negation kept; msg 2: negation LOST
        conn.execute("INSERT INTO messages VALUES (1, '决定不参与百合花定增')")
        conn.execute("INSERT INTO messages VALUES (2, '全部清仓止盈，勿留恋')")
        conn.execute("INSERT INTO facts VALUES (10, '百合花定增: 不参与', 1)")
        conn.execute("INSERT INTO facts VALUES (11, '止盈清仓完成', 2)")  # lost 勿/全部
        conn.commit()
        conn.close()
        # state db with source messages
        sdb = tmp_path / "state.db"
        sc = sqlite3.connect(sdb)
        sc.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        sc.execute("INSERT INTO messages VALUES (1, '决定不参与百合花定增')")
        sc.execute("INSERT INTO messages VALUES (2, '全部清仓止盈，勿留恋')")
        sc.commit(); sc.close()

        report = sample_audit(str(db), limit=100, state_db_path=str(sdb))
        assert report["pairs"] == 2
        assert report["negatives_retained"] == pytest.approx(0.5)
        assert report["worst_examples"], "must list loss examples"

    def test_tolerates_missing_tables(self, tmp_path):
        import sqlite3
        db = tmp_path / "b.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT)")
        conn.commit(); conn.close()
        report = sample_audit(str(db), limit=10)
        assert report["pairs"] == 0  # no msg_id column / no messages → empty report
