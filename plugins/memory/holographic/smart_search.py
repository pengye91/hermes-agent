"""Smart search: router-driven retrieval strategy for FactRetriever.

Composition of the three retrieval upgrades from the 2026-08-30 research
brief (agent-memory-retrieval-2025-2026.md):

1. ``route_query`` classifies the query → (kind, time_filter). Pure and
   free.
2. ``apply_time_filter`` prunes the over-fetched candidate pool by the
   ``[L|yymmdd]`` headers BEFORE statistical ranking (LongMemEval: +11.3%
   recall from time-aware pruning). Header-less facts always survive.
3. kind=="multihop" → ``decompose_query`` splits the query; each sub-query
   retrieves its own pool; pools merge round-robin (rank position i of
   every sub-query interleave), dedup by fact_id (ACL-SRW 2025: MRR@10
   +36.7% on multi-hop with decomposition+rerank).
4. kind=="single" → EXACTLY the existing search path, zero behavioural
   change on the fact-lookup majority path (over-split tax −5pp is real).

Failure containment: every stage degrades to plain search. No raises.
"""
from __future__ import annotations

from datetime import date
import logging
import re

from .query_router import route_query, apply_time_filter, strip_time_words
from .query_decompose import decompose_query, _DecomposeClient
from .llm_rerank import _boilerplate_strip
logger = logging.getLogger(__name__)

# Overfetch factor for pool building — pruning and merging need headroom
# above the caller's limit.
_OVERFETCH = 3

# Cross-encoder rerank depth: only the top-K of the merged pool is scored
# (~0.2s/doc CPU); deeper is latency without precision (rerank-k literature
# consensus: the knee is between 8 and 25).
_XENC_K = 8


def _subject_form(query: str) -> str:
    """Time words + conversational boilerplate stripped — the subject-only
    form that buries targets least (both lessons from the 2026-08-30 A/B:
    '今年4月…进展' and '…是什么' both drowned their subjects in noise)."""
    # question phrases FIRST, before boilerplate's word-level strip —
    # '智谱有什么进展' must lose '有什么' as a unit or the residual '有'
    # glues onto the subject ('智谱有' matches no trigram → empty pool).
    out = _QUESTION_PHRASES.sub(" ", query or "")
    out = strip_time_words(out)
    out = _boilerplate_strip(out)
    return out.strip()


_QUESTION_PHRASES = re.compile(
    r"有什么|怎么样|怎么了|如何|为什么|为何|多少|几号|哪天|何时|哪里|哪儿|什么意思")

# Content-less modifiers that carry no retrieval signal ('最新的XX动态'
# → the entity is the signal, the rest is shape).
_MODIFIER_WORDS = re.compile(r"最新|最新版|动态|情况|信息|进展|相关|最近")


def _extract_entity(subject: str) -> str | None:
    """Pull the single most entity-like CJK run from a subject form.

    Heuristic, deliberately simple: drop modifier words, strip glue chars
    from run edges ('智谱的' → '智谱'), then take the longest remaining
    run of 2-6 chars. Returns None when nothing entity-like survives.
    """
    cleaned = _MODIFIER_WORDS.sub(" ", subject)
    glue = "的了吗呢吧啊与和或者是在对于从被把给"
    runs = [
        r.strip(glue)
        for r in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", cleaned)
    ]
    runs = [r for r in runs if 2 <= len(r) <= 6 and not r.isdigit()]
    if not runs:
        return None
    return max(runs, key=len)


def smart_search(retriever, query: str, *, limit: int = 10,
                 now: date | None = None,
                 decompose_client=None,
                 xenc_reranker=None) -> list[dict]:
    """Router-driven search over a FactRetriever. Never raises past the
    underlying search; degrades to plain retrieval at every failure point.

    ``xenc_reranker``: optional CrossEncoderReranker — when available it
    re-scores the top-``_XENC_K`` of the final merged pool (bge-reranker
    v2-m3 int8 ONNX; ~0.2s/doc on 4-core CPU → ~1.6s at k=8, explicit
    searches only). Passthrough when None/unavailable/error.
    """
    plan = route_query(query, now=now)

    if plan.kind == "single":
        out = _single(retriever, query, plan, limit)
    else:
        out = _multihop(retriever, query, plan, limit, decompose_client)

    if xenc_reranker is not None:
        from .xenc_rerank import rerank_candidates
        head = out[:_XENC_K]
        tail = out[_XENC_K:]
        head = rerank_candidates(query, head, xenc_reranker)
        out = head + tail
    return out


# ---------------------------------------------------------------------------
# single path
# ---------------------------------------------------------------------------

def _single(retriever, query: str, plan, limit: int) -> list[dict]:
    if plan.time_filter is None:
        # zero-change fast path: identical call signature to plain search
        return retriever.search(query, limit=limit)

    # temporal single: strip the time words AND boilerplate (both are
    # retrieval noise — 2026-08-30 A/B: in-window facts never entered the
    # pool with them). DUAL POOL round-robin-merged: subject-form pool +
    # entity-only pool interleaved by rank ('最新版本' noise is in-window
    # too, so sequential merge lets it monopolize the head; interleave
    # guarantees entity facts surface near the top).
    subject = _subject_form(query) or query
    pool_a = retriever.search(subject, limit=limit * _OVERFETCH)
    entity = _extract_entity(subject)
    pool_b: list[dict] = []
    if entity and entity not in subject.split():
        pool_b = retriever.search(entity, limit=_OVERFETCH * 8)
    pruned_a = apply_time_filter(pool_a, plan.time_filter)
    pruned_b = apply_time_filter(pool_b, plan.time_filter)
    merged = _round_robin_merge([pruned_a, pruned_b]) if pool_b else pruned_a
    return merged[:limit]


# ---------------------------------------------------------------------------
# multihop path
# ---------------------------------------------------------------------------

def _multihop(retriever, query: str, plan, limit: int, client) -> list[dict]:
    client = client if client is not None else _DecomposeClient()
    subs = decompose_query(query, client=client)
    if len(subs) <= 1:
        # degenerate decomposition (or failure) → single-path behaviour
        return _single(retriever, query, plan, limit)

    per_sub = max(limit, 4)
    pools: list[list[dict]] = []
    for sq in subs:
        try:
            # sub-queries carry their own boilerplate ('…是什么') — strip
            # before retrieval or distractors with query-shape overlap win
            sq_subject = _subject_form(sq) or sq
            pool = retriever.search(sq_subject, limit=per_sub * _OVERFETCH)
        except Exception as e:  # sub-query must not kill the whole search
            logger.debug("sub-query search failed (%s): %s", sq, e)
            pool = []
        if plan.time_filter is not None:
            pool = apply_time_filter(pool, plan.time_filter)
        pools.append(pool[:per_sub])

    merged = _round_robin_merge(pools)
    return merged[:limit]


def _round_robin_merge(pools: list[list[dict]]) -> list[dict]:
    """Interleave pools by rank position; dedup by fact_id, first-wins.

    Rank 1 of sub-query A and rank 1 of sub-query B are both early —
    neither entity's pool dominates the merged head. This mirrors the
    complementary-evidence assembly from the decomposition literature
    (coverage first, rerank restores precision later).
    """
    seen: set = set()
    merged: list[dict] = []
    max_len = max((len(p) for p in pools), default=0)
    for i in range(max_len):
        for pool in pools:
            if i < len(pool):
                f = pool[i]
                fid = f.get("fact_id")
                if fid in seen:
                    continue
                seen.add(fid)
                merged.append(f)
    return merged
