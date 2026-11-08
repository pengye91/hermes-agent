"""Tests for the fact_store tool handler's strategy='smart' wiring.

The handler must:
- default (strategy omitted / 'off'): route through rerank_search exactly
  as today — zero behavioural change for existing callers;
- strategy='smart': route through smart_search (router+time-filter+decomp)
  BEFORE the optional llm rerank stage.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from datetime import date

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic import smart_search as smart


class _FakeMemory:
    """Minimal HolographicMemory-like object exposing the search handler
    logic without the full plugin lifecycle."""

    def __init__(self, retriever, min_trust=0.3):
        self._store = retriever.store
        self._retriever = retriever
        self._min_trust = min_trust

    handle = None  # bound below via the real handler implementation


def _seed(store: MemoryStore) -> None:
    store.add_fact("[L|260615|百合花] 百合花纯度门槛定为1.3%", category="memory")
    store.add_fact("[L|260710|百合花] 百合花纯度门槛上调至2.0%", category="memory")
    store.add_fact("[L|260720|智谱] 智谱港股代码2513.HK", category="memory")
    store.add_fact("[L|260721|MiniMax] MiniMax港股代码0100.HK", category="memory")


class TestHandlerWiring:
    def test_default_strategy_matches_smart_off_bitwise(self, tmp_path):
        """strategy omitted → identical output to the pre-smart handler."""
        store = MemoryStore(str(tmp_path / "w.db"))
        _seed(store)
        r = FactRetriever(store=store)
        from plugins.memory.holographic import HolographicMemoryProvider
        mem = HolographicMemoryProvider.__new__(HolographicMemoryProvider)
        mem._store = store
        mem._retriever = r
        mem._min_trust = 0.3
        out = json.loads(mem._handle_fact_store({"action": "search",
                                                 "query": "百合花纯度门槛"}))
        plain = json.loads(json.dumps(r.search("百合花纯度门槛", limit=10)))
        assert [f["fact_id"] for f in out["results"]] == \
               [f["fact_id"] for f in plain]
        store.close()

    def test_strategy_smart_routes_through_smart_search(self, tmp_path, monkeypatch):
        """strategy='smart' must call smart_search — pinned by monkeypatching
        it to a sentinel, ensuring the handler routes (not reimplements)."""
        store = MemoryStore(str(tmp_path / "w2.db"))
        _seed(store)
        r = FactRetriever(store=store)
        from plugins.memory.holographic import HolographicMemoryProvider
        mem = HolographicMemoryProvider.__new__(HolographicMemoryProvider)
        mem._store = store
        mem._retriever = r
        mem._min_trust = 0.3

        calls = {}

        def fake_smart(retriever, query, **kw):
            calls["query"] = query
            calls["limit"] = kw.get("limit")
            return [{"fact_id": 1, "content": "sentinel", "score": 1.0}]

        import plugins.memory.holographic.smart_search as ss
        monkeypatch.setattr(ss, "smart_search", fake_smart)

        # handler imports smart_search lazily — patch at the package level
        import plugins.memory.holographic as pkg
        monkeypatch.setattr(pkg.smart_search, "smart_search", fake_smart)

        out = json.loads(mem._handle_fact_store(
            {"action": "search", "query": "上个月百合花纯度门槛",
             "strategy": "smart"}))
        assert calls["query"] == "上个月百合花纯度门槛"
        assert out["results"][0]["content"] == "sentinel"
        store.close()
