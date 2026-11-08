"""Tests for time-window filtering of candidate facts (QueryPlan.time_filter).

Facts carry ``[L|yymmdd|主体]`` headers from the LLM distiller. The filter
keeps facts whose ANY header date falls inside the window; header-less
facts (curated imports, migrated stores) are NEVER punished — unknown date
≠ outside window. If filtering would empty the pool, return it unchanged
(graceful degradation beats empty recall).

Evidence: LongMemEval time-aware pruning +11.3% recall — but their corpus
was fully timestamped; ours is not, hence the keep-unknown policy.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.query_router import apply_time_filter


def fact(fid: str, content: str) -> dict:
    return {"fact_id": fid, "content": content, "score": 1.0}


class TestApplyTimeFilter:
    def test_keeps_fact_with_header_inside_window(self):
        pool = [fact("1", "[L|260615|百合花] 纯度门槛1.3%")]
        out = apply_time_filter(pool, ("260601", "260630"))
        assert [f["fact_id"] for f in out] == ["1"]

    def test_prunes_fact_with_header_outside_window(self):
        pool = [
            fact("1", "[L|250615|百合花] 纯度门槛1.3%"),
            fact("2", "[L|260620|百合花] 上调到2%"),
        ]
        out = apply_time_filter(pool, ("260601", "260630"))
        assert [f["fact_id"] for f in out] == ["2"]

    def test_headerless_fact_is_never_pruned(self):
        pool = [
            fact("1", "玻璃基板加工端位于价值链最差位置"),
            fact("2", "[L|250615|百合花] 纯度门槛"),
        ]
        out = apply_time_filter(pool, ("260601", "260630"))
        assert [f["fact_id"] for f in out] == ["1"]

    def test_any_header_date_in_window_suffices(self):
        # fact mentions two dates; the later one is in-window → keep.
        pool = [fact("1", "[L|250101|澜起] 首次调研 … [L|260702|澜起] 中报验证")]
        out = apply_time_filter(pool, ("260601", "260830"))
        assert [f["fact_id"] for f in out] == ["1"]

    def test_all_outside_window_returns_pool_unchanged(self):
        # graceful degradation: prune-to-empty → return original pool
        pool = [fact("1", "[L|250615|百合花] 纯度门槛")]
        out = apply_time_filter(pool, ("260601", "260630"))
        assert [f["fact_id"] for f in out] == ["1"]

    def test_empty_pool_and_none_filter_passthrough(self):
        assert apply_time_filter([], ("260101", "260830")) == []
        pool = [fact("1", "anything")]
        assert apply_time_filter(pool, None) is pool

    def test_boundary_dates_are_inclusive(self):
        pool = [
            fact("1", "[L|260601|a] start boundary"),
            fact("2", "[L|260630|b] end boundary"),
            fact("3", "[L|260602|c] interior"),
        ]
        out = apply_time_filter(pool, ("260601", "260630"))
        assert [f["fact_id"] for f in out] == ["1", "2", "3"]

    def test_mixed_pool_keeps_headerless_plus_inwindow(self):
        pool = [
            fact("1", "无头事实"),
            fact("2", "[L|260510|x] 窗口外"),
            fact("3", "[L|260710|y] 窗口内"),
        ]
        out = apply_time_filter(pool, ("260601", "260830"))
        assert [f["fact_id"] for f in out] == ["1", "3"]

    def test_preserves_order_and_other_fields(self):
        pool = [fact("1", "[L|260701|a] x"), fact("2", "[L|260702|b] y")]
        pool[0]["score"], pool[1]["score"] = 0.9, 0.5
        out = apply_time_filter(pool, ("260601", "260830"))
        assert [f["score"] for f in out] == [0.9, 0.5]
