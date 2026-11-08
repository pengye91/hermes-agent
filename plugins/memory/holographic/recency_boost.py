"""Soft-recency boost — additive, never punitive (cycle 7).

The pre-existing temporal_decay (half-life) is a MULTIPLICATIVE penalty:
0.5^(age/half_life) sinks old facts toward zero. That is the wrong shape
for this corpus — the 2026-08-30 embedding-dedup audit measured that
old/high-similarity pairs are often legitimate CONCLUSION EVOLUTION
(same entity, months apart, opposite verdicts) which the memory exists
to keep. Research-brief §Top-5 #5 (Relay's gradient time-window) calls
for soft gradient weighting instead.

Shape: boost = 1 + strength * 2^(-age_days / half_life)  ∈ (1, 1+strength]
- fresh fact → ×(1+strength) at most (default +25%)
- ancient fact → ×1.0 EXACTLY (no penalty, ever)
- a 2x more relevant old fact always beats a fresh one at default 0.25
"""
from __future__ import annotations

import math
import time


def recency_boost(strength: float, now_ts: float, created_at,
                  half_life_days: float = 30.0) -> float:
    """Multiplicative boost factor ∈ (1.0, 1.0+strength]. Never < 1.

    Missing/unparseable timestamps → 1.0 (old stores must not be
    systematically punished for missing metadata).
    """
    if strength <= 0:
        return 1.0
    try:
        ts = float(created_at)
    except (TypeError, ValueError):
        return 1.0
    if ts <= 0:
        return 1.0
    age_days = max(0.0, (now_ts - ts) / 86400.0)
    return 1.0 + strength * math.pow(0.5, age_days / max(1e-9, half_life_days))


def apply_recency_boost(pool: list[dict], strength: float,
                        half_life_days: float, now_ts: float | None = None,
                        ) -> list[dict]:
    """Return a NEW list of dicts with score × boost (score kept when the
    fact carries none — boosting needs a base). Input not mutated."""
    if now_ts is None:
        now_ts = time.time()
    out = []
    for f in pool:
        g = dict(f)
        if g.get("score") is not None:
            g["score"] = g["score"] * recency_boost(
                strength, now_ts, g.get("created_at"), half_life_days)
        out.append(g)
    return out
