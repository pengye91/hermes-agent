"""Tests for the query-type router (FactRetriever retrieval strategy selector).

Evidence base (2026-08-30 research brief):
- LongMemEval: time-aware query expansion +11.3% recall (round granularity).
- ACL-SRW 2025: multi-hop decomposition MRR@10 +36.7%, but -5pp on
  fact-lookup (over-split tax) — hence routing, not global rewrite.
- DeCoR: HyDE/Query2Doc negative on multi-hop corpora — excluded by design.

The router classifies a query into a retrieval strategy + optional time
window. It must be pure (no LLM, no DB) so it stays testable and free.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from datetime import date

from plugins.memory.holographic.query_router import (
    QueryPlan,
    route_query,
)


NOW = date(2026, 8, 30)


class TestKindClassification:
    def test_plain_fact_lookup_is_single(self):
        plan = route_query("百合花的纯度门槛是多少", now=NOW)
        assert plan.kind == "single"
        assert plan.sub_queries == []

    def test_conclusion_question_is_single_not_multihop(self):
        # "结论是什么" is a fact-lookup about one subject — must NOT be
        # routed to decomposition (the over-split tax).
        plan = route_query("我们当时对百合花的结论是什么", now=NOW)
        assert plan.kind == "single"

    def test_enum_marker_routes_to_multihop(self):
        plan = route_query("智谱和MiniMax的港股代码分别是什么", now=NOW)
        assert plan.kind == "multihop"

    def test_comparison_marker_routes_to_multihop(self):
        plan = route_query("智谱和MiniMax哪个更值得买", now=NOW)
        assert plan.kind == "multihop"

    def test_bare_conjunction_stays_single(self):
        # "和" alone (no enum/comparison marker) is a single topic query —
        # over-triggering decomposition is the documented failure mode.
        plan = route_query("玻璃基板和设备的调研框架", now=NOW)
        assert plan.kind == "single"


class TestTimeFilter:
    def test_no_time_words_no_filter(self):
        plan = route_query("百合花的纯度门槛", now=NOW)
        assert plan.time_filter is None
        assert plan.is_temporal is False

    def test_this_year(self):
        plan = route_query("今年我们调研了哪些玻璃基板标的", now=NOW)
        assert plan.is_temporal is True
        assert plan.time_filter == ("260101", "260830")

    def test_last_year(self):
        plan = route_query("去年买过什么存储标的", now=NOW)
        assert plan.time_filter == ("250101", "251231")

    def test_last_month(self):
        plan = route_query("上个月的复盘结论是什么", now=NOW)
        assert plan.time_filter == ("260701", "260731")

    def test_recent_n_days(self):
        plan = route_query("最近14天的持仓变动", now=NOW)
        # inclusive 14-day window ending 0830 → starts 0817 (14 days total)
        assert plan.time_filter == ("260817", "260830")

    def test_explicit_year_month(self):
        plan = route_query("2026年6月我们买入了什么", now=NOW)
        assert plan.time_filter == ("260601", "260630")

    def test_explicit_year_alone(self):
        plan = route_query("2025年有哪些涨停复盘", now=NOW)
        assert plan.time_filter == ("250101", "251231")

    def test_bare_month_resolves_to_most_recent_past(self):
        # "3月" with now=2026-08-30 → March 2026 (most recent past March)
        plan = route_query("3月讨论的结论是什么", now=NOW)
        assert plan.time_filter == ("260301", "260331")

    def test_bare_future_month_resolves_to_last_year(self):
        # "12月" with now=2026-08-30 → December 2025
        plan = route_query("12月的年度总结", now=NOW)
        assert plan.time_filter == ("251201", "251231")

    def test_half_year_marker(self):
        plan = route_query("上半年的组合调整", now=NOW)
        assert plan.time_filter == ("260101", "260630")


class TestOrthogonality:
    def test_temporal_and_multihop_compose(self):
        # kind (retrieval strategy) and time_filter are orthogonal dims.
        plan = route_query("去年智谱和MiniMax谁涨幅更大", now=NOW)
        assert plan.kind == "multihop"
        assert plan.time_filter == ("250101", "251231")

    def test_plan_carries_original_query(self):
        plan = route_query("百合花 纯度", now=NOW)
        assert plan.query == "百合花 纯度"


class TestRobustness:
    def test_empty_query_never_raises(self):
        plan = route_query("", now=NOW)
        assert plan.kind == "single"
        assert plan.time_filter is None

    def test_default_now_used_when_omitted(self):
        plan = route_query("今年调研的标的")  # no `now` → system date
        assert plan.is_temporal is True

    def test_query_plan_is_frozen_dataclass(self):
        plan = route_query("x", now=NOW)
        with pytest.raises(Exception):
            plan.kind = "mutated"


class TestBoundaryFormat:
    def test_filter_bounds_are_six_char_yymmdd_strings(self):
        # facts carry `[L|yymmdd]` headers — filter must match that format.
        plan = route_query("2026年6月的买入", now=NOW)
        start, end = plan.time_filter
        assert len(start) == 6 and len(end) == 6
        assert start.isdigit() and end.isdigit()
        assert start <= end
