"""Tests for multi-hop query decomposition (glm-5.3-flash, mocked LLM).

Evidence: ACL-SRW 2025 — decomposition MRR@10 +36.7% on multi-hop, but
-5pp on fact-lookup; the router only sends kind=="multihop" queries here.
Anti-fabrication rules mirror llm_rerank: sub-queries must be validated
strings; failure → [original query] (single-path degradation).
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.query_decompose import (
    decompose_query,
    _parse_sub_queries,
)


class TestParseSubQueries:
    def test_plain_json_array(self):
        assert _parse_sub_queries('["智谱的港股代码", "MiniMax的港股代码"]') == [
            "智谱的港股代码", "MiniMax的港股代码"]

    def test_fenced_json_array(self):
        raw = "```json\n[\"子查询1\", \"子查询2\"]\n```"
        assert _parse_sub_queries(raw) == ["子查询1", "子查询2"]

    def test_prose_wrapped_array_is_extracted(self):
        raw = "好的,分解如下: [\"查询甲\", \"查询乙\"] 以上。"
        assert _parse_sub_queries(raw) == ["查询甲", "查询乙"]

    def test_garbage_returns_empty(self):
        assert _parse_sub_queries("这不是JSON") == []
        assert _parse_sub_queries("") == []
        assert _parse_sub_queries(None) == []

    def test_non_string_items_dropped(self):
        assert _parse_sub_queries('["好的q", 42, null, "good"]') == ["好的q", "good"]

    def test_too_short_items_dropped(self):
        assert _parse_sub_queries('["ab", "abc"]') == ["abc"]

    def test_duplicates_dropped(self):
        assert _parse_sub_queries('["查询1", "查询1", "查询2"]') == ["查询1", "查询2"]

    def test_overlong_item_truncated_or_dropped(self):
        # >200-char sub-query is garbage (model rambling) — must not pass raw
        long_item = "超长" * 150
        out = _parse_sub_queries(json_dumps([long_item, "正常查询"]))
        assert all(len(s) <= 200 for s in out)
        assert "正常查询" in out

    def test_single_item_is_valid(self):
        assert _parse_sub_queries('["只有一个"]') == ["只有一个"]


def json_dumps(x) -> str:
    import json as _json
    return _json.dumps(x, ensure_ascii=False)


class TestDecomposeQuery:
    def test_returns_original_when_llm_unavailable(self):
        class Dead:
            available = False
        out = decompose_query("智谱和MiniMax的代码分别是什么", client=Dead())
        assert out == ["智谱和MiniMax的代码分别是什么"]

    def test_returns_original_on_llm_failure(self):
        class Exploding:
            available = True
            def decompose(self, query, max_subs):
                raise RuntimeError("HTTP 429")
        out = decompose_query("q", client=Exploding())
        assert out == ["q"]

    def test_returns_original_on_empty_llm_output(self):
        class Empty:
            available = True
            def decompose(self, query, max_subs):
                return "垃圾输出"
        out = decompose_query("q", client=Empty())
        assert out == ["q"]

    def test_valid_subqueries_pass_through(self):
        class Ok:
            available = True
            def decompose(self, query, max_subs):
                return '["智谱的港股代码", "MiniMax的港股代码"]'
        out = decompose_query("智谱和MiniMax的代码分别是什么", client=Ok())
        assert out == ["智谱的港股代码", "MiniMax的港股代码"]

    def test_caps_subquery_count(self):
        class Many:
            available = True
            def decompose(self, query, max_subs):
                return json_dumps([f"子查询{i}" for i in range(10)])
        out = decompose_query("q", client=Many())
        assert 1 <= len(out) <= 4  # max_subs default cap

    def test_client_none_degrades_to_original(self):
        out = decompose_query("任何查询", client=None)
        assert out == ["任何查询"]

    def test_empty_query_degrades_to_empty_list(self):
        class Ok:
            available = True
            def decompose(self, query, max_subs):
                return '["never"]'
        out = decompose_query("", client=Ok())
        assert out == []
