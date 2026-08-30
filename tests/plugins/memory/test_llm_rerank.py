"""TDD tests for the optional LLM listwise rerank on fact_store search (Fix #3).

Statistical rerank tops out at lexical/statistical signals; queries like
"我们当时对X的结论" need judgment (conclusion vs evidence vs history).
Design: an OPTIONAL listwise rerank stage — fact_store search gains
rerank="llm" (default "off"). It takes the top-N (12) statistical
candidates, asks glm-5.3-flash (Coding Plan endpoint, thinking disabled —
same client pattern as session_distill/CLS) to pick and order the K most
relevant, and merges: reranked items keep their vsim/fts_rank fields and
gain rerank_pos + llm_rerank marker. LLM failure → original order (silent
degrade, cron-free path). Never used in per-turn prefetch (explicit calls
only).

Contract:
1. rerank="off" (default) → no LLM call, original order.
2. rerank="llm" with a working client → order follows the LLM's picks;
   items gain rerank_pos.
3. LLM failure/timeout → results returned in original order, no raise.
4. Malformed LLM output (not a JSON id list) → original order, no raise.
5. LLM may only reorder/shorten from the candidate set — no fabrication
   (unknown ids in its answer are ignored).
"""

import json
import sys
from unittest import mock

import pytest

HERMES_ROOT = "/usr/local/lib/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

from plugins.memory.holographic import llm_rerank  # noqa: E402


CANDS = [
    {"fact_id": 1, "content": "百合花投资结论：不参与", "score": 0.30},
    {"fact_id": 2, "content": "百合花颜料业务占比89.4%", "score": 0.29},
    {"fact_id": 3, "content": "飞书推送架构定局", "score": 0.28},
    {"fact_id": 4, "content": "gateway排水机制源码分析", "score": 0.27},
]


# ── pure helpers ────────────────────────────────────────

def test_parse_llm_picks_valid():
    assert llm_rerank._parse_picks('```json\n[3, 1]\n```') == [3, 1]
    assert llm_rerank._parse_picks("[1,2]") == [1, 2]


def test_boilerplate_strip_unseparated_cjk():
    """中文无分隔查询: '我们当时对百合花的结论是什么' → '百合花'。"""
    assert llm_rerank._boilerplate_strip("我们当时对百合花的结论是什么") == "百合花"


def test_boilerplate_strip_keeps_subject():
    assert llm_rerank._boilerplate_strip("止盈规则是什么") == "止盈规则"
    assert llm_rerank._boilerplate_strip("gateway重启的排水机制是怎样的") != ""


def test_boilerplate_strip_all_boilerplate():
    """全模板词查询 → 空串 (调用方据此跳过第二路取数)。"""
    assert llm_rerank._boilerplate_strip("我们当时讨论过什么") == ""


def test_parse_llm_picks_garbage():
    assert llm_rerank._parse_picks("我觉得都不相关") == []
    assert llm_rerank._parse_picks("") == []
    assert llm_rerank._parse_picks(None) == []


def test_apply_picks_reorders_and_filters():
    out = llm_rerank._apply_picks(CANDS, [2, 1])
    assert [f["fact_id"] for f in out] == [2, 1]
    assert out[0]["rerank_pos"] == 1
    assert "llm_rerank" in out[0]


def test_apply_picks_ignores_unknown_ids():
    out = llm_rerank._apply_picks(CANDS, [2, 999])
    assert [f["fact_id"] for f in out] == [2]


def test_apply_picks_empty_returns_empty():
    assert llm_rerank._apply_picks(CANDS, []) == []


# ── rerank_search integration ───────────────────────────

def _ok_client(picks):
    class _C:
        available = True
        def rerank(self, query, items, k):
            return picks
    return _C()


def test_off_no_llm_call():
    calls = {"n": 0}
    class _C:
        available = True
        def rerank(self, *a, **k):
            calls["n"] += 1
            return []
    out = llm_rerank.rerank_search(
        lambda q, limit: list(CANDS), "百合花 结论", _C(), rerank="off")
    assert calls["n"] == 0
    assert [f["fact_id"] for f in out] == [1, 2, 3, 4]


def test_llm_order_followed():
    out = llm_rerank.rerank_search(
        lambda q, limit: list(CANDS), "百合花 结论", _ok_client([2, 1]),
        rerank="llm", k=2)
    assert [f["fact_id"] for f in out] == [2, 1]
    assert out[0]["rerank_pos"] == 1


def test_llm_failure_original_order():
    class _Boom:
        available = True
        def rerank(self, *a, **k):
            raise TimeoutError("llm down")
    out = llm_rerank.rerank_search(
        lambda q, limit: list(CANDS), "q", _Boom(), rerank="llm")
    assert [f["fact_id"] for f in out] == [1, 2, 3, 4]
    assert "rerank_pos" not in out[0]


def test_unavailable_client_original_order():
    class _None:
        available = False
    out = llm_rerank.rerank_search(
        lambda q, limit: list(CANDS), "q", _None(), rerank="llm")
    assert [f["fact_id"] for f in out] == [1, 2, 3, 4]


def test_no_client_original_order():
    out = llm_rerank.rerank_search(
        lambda q, limit: list(CANDS), "q", None, rerank="llm")
    assert [f["fact_id"] for f in out] == [1, 2, 3, 4]
