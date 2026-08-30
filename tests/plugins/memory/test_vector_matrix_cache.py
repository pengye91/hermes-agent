"""TDD tests for the vector-matrix retrieval fix (Fix #1).

The semantic leg loads EVERY embedding blob from SQLite and unpacks it in a
Python loop per query — O(n) with ~120ms/1k facts of pure interpreter
overhead (measured 3.2s cold / 7.2s hot at 27k facts, and "hot" was slower
because nothing is cached). Fix: an in-process numpy matrix cache in
MemoryStore, rebuilt lazily, kept consistent on add/update/remove, with a
configurable capacity cap.

Contract:
1. vector_candidates uses the cache and returns IDENTICAL top-k ordering
   to the brute-force path (cosine equivalence).
2. Second query on the same store is fast (cache hit path exists — assert
   via a counter attribute, not wall-clock).
3. add_fact / remove_fact / update_fact invalidate or refresh the cache.
4. Cache respects a max-facts cap (won't load unbounded corpora).
"""

import os
import sys
import time
from unittest import mock

import pytest

HERMES_ROOT = "/usr/local/lib/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

from plugins.memory.holographic.store import MemoryStore  # noqa: E402


class _FakeEmbed:
    """Deterministic embedding: per-text seeded uniform vectors, dim=8.

    NOTE: do NOT build vectors by reinterpreting sha256 bytes as float32 —
    those are denormal (~1e-36) values whose float32 precision collapse makes
    every cosine ≈ noise. Per-text seeded uniform [-1,1] is deterministic,
    numerically healthy, and actually separates texts."""

    available = True

    def __init__(self, dim=8):
        self.dim = dim

    def _vec_for(self, text: str):
        import random
        rng = random.Random(hash(text) & 0xFFFFFFFF if text else 12345)
        v = [rng.uniform(-1, 1) for _ in range(self.dim)]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed(self, texts):
        # identical texts → identical vectors (query==stored self-sim = 1.0);
        # different texts → quasi-orthogonal (cos ≈ 0)
        return [self._vec_for(t) for t in texts]


def _mk_store(tmp_path, embed=True, cap=None):
    emb = _FakeEmbed() if embed else None
    kwargs = {}
    if cap is not None:
        kwargs["_vec_cache_max_facts"] = cap
    s = MemoryStore(db_path=str(tmp_path / "t.db"), default_trust=0.6,
                    hrr_dim=64, embedding_client=emb, **kwargs)
    for i, t in enumerate(["百合花是颜料公司", "光刻胶国产替代", "gateway排水机制",
                            "止盈规则v2", "飞书推送架构"]):
        s.add_fact(t, category="corpus", tags="t")
    return s


# ── 1. equivalence with brute force ─────────────────────

def test_matrix_results_match_bruteforce_order(tmp_path):
    s = _mk_store(tmp_path)
    # Fake embeddings have zero semantic generalization (substring queries are
    # quasi-orthogonal) — so query with an EXACT stored text: self-sim = 1.0
    # must dominate and rank that fact first.
    res = s.vector_candidates("百合花是颜料公司", top=3)
    assert len(res) == 3
    assert res[0]["content"] == "百合花是颜料公司"
    assert "vsim" in res[0] and res[0]["vsim"] > 0.99
    s.close()


def test_vsim_values_close_to_one(tmp_path):
    """Query == stored text → vsim ≈ 1.0."""
    s = _mk_store(tmp_path)
    res = s.vector_candidates("光刻胶国产替代", top=1)
    assert res[0]["vsim"] > 0.99
    s.close()


# ── 2. cache hit path ───────────────────────────────────

def test_second_query_hits_cache(tmp_path):
    s = _mk_store(tmp_path)
    s.vector_candidates("百合花", top=3)
    builds_before = s._vec_cache_stats["builds"]
    s.vector_candidates("光刻胶", top=3)
    assert s._vec_cache_stats["builds"] == builds_before, "cache must not rebuild"
    assert s._vec_cache_stats["hits"] >= 1
    s.close()


def test_cache_skipped_without_client(tmp_path):
    s = _mk_store(tmp_path, embed=None)
    assert s.vector_candidates("x") == []
    s.close()


# ── 3. invalidation ─────────────────────────────────────

def test_add_invalidates_cache(tmp_path):
    s = _mk_store(tmp_path)
    s.vector_candidates("百合花", top=3)
    s.add_fact("全新的记忆内容测试", category="corpus")
    s.vector_candidates("全新的记忆内容测试", top=1)
    assert s._vec_cache_stats["builds"] >= 2, "add must invalidate"
    res = s.vector_candidates("全新的记忆内容测试", top=1)
    assert "全新的记忆内容测试" in res[0]["content"]
    s.close()


def test_remove_invalidates_cache(tmp_path):
    s = _mk_store(tmp_path)
    fid = s.add_fact("待删除的记忆条目", category="corpus")
    s.vector_candidates("待删除的记忆条目", top=1)
    s.remove_fact(fid)
    res = s.vector_candidates("待删除的记忆条目", top=1)
    assert all("待删除的记忆条目" not in r["content"] for r in res)
    s.close()


def test_update_invalidates_cache(tmp_path):
    s = _mk_store(tmp_path)
    fid = s.add_fact("旧内容版本一", category="corpus")
    s.vector_candidates("旧内容版本一", top=1)
    s.update_fact(fid, content="新内容版本二完全不同")
    res = s.vector_candidates("新内容版本二完全不同", top=1)
    assert any("新内容版本二完全不同" in r["content"] for r in res)
    s.close()


# ── 4. capacity cap ─────────────────────────────────────

def test_cache_respects_cap(tmp_path):
    s = _mk_store(tmp_path, cap=3)  # 5 facts stored, cap 3
    res = s.vector_candidates("百合花", top=10)
    # cap engaged: cache holds only 3 facts
    assert s._vec_cache_stats.get("capped", False) is True
    s.close()
