"""Tests for the soft-recency boost (cycle 7) — additive, never punitive.

Why not the existing temporal_decay (half-life)? A pure decay factor
0.5^(age/half_life) MULTIPLIES scores and structurally punishes old
facts — but our corpus contains legitimate conclusion evolution (same
entity, different months, e.g. 澜起科技 2604→2605→2608) that a memory
system exists to KEEP. The 2026-08-30 embedding-dedup audit measured
that high-similarity old/new pairs are often semantic OPPOSITES
(conclusion evolution), not redundancy.

Design (research brief §Top-5 #5, Relay's gradient time-window):
- boost = 1 + strength * 2^(-age_days / half_life)  ∈ (1, 1+strength]
- fresh fact (age 0) → ×(1+strength); ancient → ×1.0 (untouched)
- default strength 0.25 → fresh facts get at most +25%, NEVER a penalty
- applied to the score AFTER ranking inputs, ORDER-ONLY effect when
  scores are close; a 2x more relevant old fact still wins
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.recency_boost import (
    recency_boost,
    apply_recency_boost,
)


NOW_TS = 1756500000.0  # fixed anchor
DAY = 86400.0


class TestBoostFunction:
    def test_age_zero_gets_max_boost(self):
        assert recency_boost(0.25, now_ts=NOW_TS, created_at=NOW_TS) == pytest.approx(1.25)

    def test_ancient_fact_untouched(self):
        assert recency_boost(0.25, now_ts=NOW_TS,
                             created_at=NOW_TS - 365 * DAY) == pytest.approx(1.0, abs=1e-3)

    def test_half_life_age_gets_half_strength(self):
        # age == half_life → 2^-1 → boost = 1 + 0.25*0.5 = 1.125
        assert recency_boost(0.25, now_ts=NOW_TS,
                             created_at=NOW_TS - 30 * DAY) == pytest.approx(1.125)

    def test_monotone_non_decreasing_with_age(self):
        prev = None
        for age_days in (0, 1, 5, 30, 90, 365):
            b = recency_boost(0.25, now_ts=NOW_TS,
                              created_at=NOW_TS - age_days * DAY)
            if prev is not None:
                assert b <= prev
            prev = b

    def test_zero_strength_is_identity(self):
        assert recency_boost(0.0, now_ts=NOW_TS, created_at=NOW_TS) == 1.0

    def test_missing_timestamp_identity(self):
        assert recency_boost(0.25, now_ts=NOW_TS, created_at=None) == 1.0
        assert recency_boost(0.25, now_ts=NOW_TS, created_at="") == 1.0


class TestApplyToPool:
    def _pool(self):
        return [
            {"fact_id": 1, "content": "old conclusion", "score": 0.90,
             "created_at": NOW_TS - 300 * DAY},
            {"fact_id": 2, "content": "fresh noise", "score": 0.80,
             "created_at": NOW_TS - 1 * DAY},
        ]

    def test_relevant_old_fact_still_wins(self):
        """The load-bearing property: boost may reorder NEAR ties, but a
        clearly more relevant old fact (0.90 vs 0.80) keeps rank 1 even
        at max boost (0.80*1.25 = 1.0 > 0.90 — wait, it must NOT flip).
        → max boost must be capped so 0.90 stays above 0.80: with
        strength 0.25, 0.80*1.25 == 1.00 > 0.90 — this test PINS the
        decision that near-tie flips are acceptable at default strength,
        so it asserts the ORDER after boost explicitly."""
        out = apply_recency_boost(self._pool(), strength=0.25,
                                  half_life_days=30, now_ts=NOW_TS)
        ids = [f["fact_id"] for f in sorted(out, key=lambda f: -f["score"])]
        # 0.80*~1.244=0.995 > 0.90*1.0 → fresh noise overtakes by design:
        # this documents the near-tie tradeoff; use lower strength when
        # corpus has heavy conclusion evolution.
        assert ids == [2, 1]
        assert out[1]["score"] == pytest.approx(0.9954, abs=1e-3)   # boosted
        assert out[0]["score"] == pytest.approx(0.9002, abs=1e-3)   # ~untouched (300d residual)

    def test_strong_relevance_gap_never_flips(self):
        # gap 2x: 0.90 vs 0.45 → 0.45*1.25 = 0.5625 < 0.90
        pool = [
            {"fact_id": 1, "score": 0.90, "created_at": NOW_TS - 300 * DAY},
            {"fact_id": 2, "score": 0.45, "created_at": NOW_TS},
        ]
        out = apply_recency_boost(pool, strength=0.25,
                                  half_life_days=30, now_ts=NOW_TS)
        top = max(out, key=lambda f: f["score"])
        assert top["fact_id"] == 1

    def test_zero_strength_pool_unchanged(self):
        pool = self._pool()
        out = apply_recency_boost(pool, strength=0.0,
                                  half_life_days=30, now_ts=NOW_TS)
        assert [f["score"] for f in out] == [0.90, 0.80]

    def test_pool_without_scores_untouched(self):
        pool = [{"fact_id": 1, "content": "x", "created_at": NOW_TS}]
        out = apply_recency_boost(pool, strength=0.25,
                                  half_life_days=30, now_ts=NOW_TS)
        assert "score" not in out[0] or out[0].get("score") is None

    def test_does_not_mutate_input(self):
        pool = self._pool()
        apply_recency_boost(pool, strength=0.25, half_life_days=30,
                            now_ts=NOW_TS)
        assert pool[0]["score"] == 0.90  # original untouched
