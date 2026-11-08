"""Tests for temporal-query retrieval with time words stripped.

Root cause found in the 2026-08-30 A/B (bench_smart_ab.py): '今年4月智谱
有什么进展' — the 2604 智谱 facts NEVER entered the top-30 candidate pool
because the time words dilute the FTS/vector signal. Pool-level filtering
cannot rescue facts that never got retrieved.

Fix: strip the time phrase from the query BEFORE retrieval (the window
filter carries the temporal intent), mirroring llm_rerank's
boilerplate-strip lesson: query boilerplate IS the noise.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.query_router import strip_time_words


class TestStripTimeWords:
    def test_strips_relative_year_month(self):
        assert strip_time_words("今年4月智谱有什么进展") == "智谱有什么进展"

    def test_strips_bare_month(self):
        assert strip_time_words("5月我们讨论的澜起科技的结论") == "我们讨论的澜起科技的结论"

    def test_strips_recent_days(self):
        assert strip_time_words("最近14天的持仓变动") == "的持仓变动"

    def test_strips_absolute_year(self):
        assert strip_time_words("2025年有哪些涨停复盘") == "有哪些涨停复盘"

    def test_strips_year_month_combo(self):
        assert strip_time_words("2026年6月我们买入了什么") == "我们买入了什么"

    def test_strips_upper_lower_half(self):
        assert strip_time_words("上半年的组合调整") == "的组合调整"
        assert strip_time_words("8月智谱的最新动态") == "智谱的最新动态"

    def test_no_time_words_unchanged(self):
        assert strip_time_words("百合花纯度门槛") == "百合花纯度门槛"

    def test_empty_and_none_safe(self):
        assert strip_time_words("") == ""
        assert strip_time_words(None) == ""

    def test_whitespace_collapsed(self):
        out = strip_time_words("今年 4月 智谱 进展")
        assert out == "智谱 进展"
