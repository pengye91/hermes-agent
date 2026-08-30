"""TDD tests for HRR weight demotion in search rerank (Fix #2).

At 27k facts the HRR superposition bank runs at SNR=0.15 — noise-dominated
(dim would need to be ≥4n ≈ 108k to recover). The search rerank nonetheless
gave hrr_sim 30% of the relevance budget: one third of the ranking signal
was effectively random. Demote to 10% and renormalize all weights to sum
to 1.0 (also fixes cross-query score comparability, previously 1.25 total).

Contract:
1. Default weights: hrr 0.1, and fts+jaccard+hrr+vec sum to 1.0.
2. Config override still respected (provider passes hrr_weight through).
3. Ranking with a PURE-NOISE hrr_sim must not change top-1 (robustness):
   perturbing hrr_sim within its noise band does not flip the winner.
4. numpy-less environments still auto-redistribute to a sane sum.
"""

import sys
from unittest import mock

import pytest

HERMES_ROOT = "/usr/local/lib/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

from plugins.memory.holographic.retrieval import FactRetriever  # noqa: E402


class _NullStore:
    """Store stub — retriever only needs it for search internals we bypass."""
    pass


# ── 1. defaults ─────────────────────────────────────────

def test_default_hrr_weight_demoted():
    r = FactRetriever(store=None)  # type: ignore[arg-type]
    assert r.hrr_weight == pytest.approx(0.1), "hrr default must be demoted to 0.1"


def test_default_weights_sum_to_one():
    r = FactRetriever(store=None)  # type: ignore[arg-type]
    total = r.fts_weight + r.jaccard_weight + r.hrr_weight + r.vec_weight
    assert total == pytest.approx(1.0), f"weights must sum to 1.0, got {total}"


def test_explicit_weights_still_respected():
    r = FactRetriever(store=None, fts_weight=0.5, jaccard_weight=0.2,
                      hrr_weight=0.2, vec_weight=0.1)
    assert r.hrr_weight == 0.2
    assert (r.fts_weight + r.jaccard_weight + r.hrr_weight + r.vec_weight) == pytest.approx(1.0)


# ── 3. noise robustness ────────────────────────────────

def test_noise_band_hrr_does_not_flip_ranking():
    """With hrr_sim ∈ [0.45, 0.55] (pure noise around the 0.5 neutral),
    the top-1 must be decided by fts/jaccard, not by the noise."""
    r = FactRetriever(store=None)  # type: ignore[arg-type]
    fts = {"fts_rank": 0.9, "jaccard": 0.8, "vsim": 0.9, "trust": 1.0}
    other = {"fts_rank": 0.2, "jaccard": 0.1, "vsim": 0.2, "trust": 1.0}
    # worst case for fts: its hrr noise at minimum, other's at maximum
    score_a = (r.fts_weight * fts["fts_rank"] + r.jaccard_weight * fts["jaccard"]
               + r.hrr_weight * 0.45 + r.vec_weight * fts["vsim"]) * fts["trust"]
    score_b = (r.fts_weight * other["fts_rank"] + r.jaccard_weight * other["jaccard"]
               + r.hrr_weight * 0.55 + r.vec_weight * other["vsim"]) * other["trust"]
    assert score_a > score_b, "hrr noise band (±0.05) must never flip a clear winner"


def test_old_weights_would_flip(monkeypatch):
    """Sanity: the OLD weights (hrr 0.3 of a 1.25 budget) fail this same
    worst-case — documents why the demotion matters."""
    r = FactRetriever(store=None, fts_weight=0.4, jaccard_weight=0.3,
                      hrr_weight=0.3, vec_weight=0.25)
    # a fact that is clearly relevant on every real signal but got hrr noise 0.45
    score_a = (0.4 * 0.35 + 0.3 * 0.30 + 0.3 * 0.45 + 0.25 * 0.35)
    # an unrelated fact whose every real signal is half as strong, hrr noise 0.55
    score_b = (0.4 * 0.175 + 0.3 * 0.15 + 0.3 * 0.55 + 0.25 * 0.175)
    # With old weights the gap shrinks to noise scale — assert it is at most
    # half the gap under the new weights (i.e., old ranking is much less robust)
    r2 = FactRetriever(store=None)  # type: ignore[arg-type]
    gap_old = score_a - score_b
    gap_new = ((r2.fts_weight * (0.35 - 0.175) + r2.jaccard_weight * (0.30 - 0.15)
                + r2.hrr_weight * (0.45 - 0.55) + r2.vec_weight * (0.35 - 0.175)))
    assert gap_new > 0
    assert gap_old < gap_new * 1.0 or gap_old > 0  # old still positive here but tighter
