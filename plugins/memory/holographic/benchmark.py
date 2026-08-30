"""Persistent memory-retrieval benchmark harness (LongMemEval port).

2026-08-29 ran an in-session throwaway benchmark on a 764-fact corpus
(references/benchmark-methodology.md). This module makes the harness a
permanent part of the holographic plugin so every future patch re-runs it
(methodology rule 6: the benchmark exists to find bugs the suite misses).

Design (from the skill's benchmark protocol):
- Corpus questions: stratified (per-month proportional) sample of corpus
  facts; judgment by EVIDENCE KEYS (any-key OR, space-normalized
  substring) — never answer-substring (paraphrase breaks it).
- Manual anchor questions: user-known facts, keys from the fact itself.
- Contenders: (name, search_fn(query_dict, limit) -> list[dict]).
- Metrics: R@1/3/5/10, MRR, latency; grouped reports.
- Benchmark artifacts (this module's own questions landing back in the
  corpus via distillation) are excluded from judgment by content marker.
"""

from __future__ import annotations

import random
import re
import time
from typing import Callable

_BENCH_MARKER = "benchmark 问题"


def stratified_corpus_sample(facts: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Per-month proportional stratified sample of corpus facts.

    Dates come from the [L|yymmdd] header (positional: chars 3..9).
    Every month present in the population gets at least one sample;
    remaining quota is allocated proportionally to month population.
    """
    by_month: dict[str, list[dict]] = {}
    for f in facts:
        c = f.get("content", "") or ""
        m = c[3:9] if len(c) > 9 and c.startswith("[L|") else "unknown"
        by_month.setdefault(m, []).append(f)
    if not by_month or n <= 0:
        return []
    rng = random.Random(seed)
    months = list(by_month.keys())
    total = sum(len(v) for v in by_month.values())
    # floor: every month gets 1 (if quota allows), rest proportional
    quota: dict[str, int] = {m: 1 for m in months}
    remaining = n - len(months)
    if remaining > 0:
        for m in sorted(months, key=lambda x: -len(by_month[x])):
            extra = min(len(by_month[m]), max(1, round(remaining * len(by_month[m]) / total)))
            quota[m] += extra
    # trim to n exactly
    out: list[dict] = []
    for m in months:
        pool = list(by_month[m])
        rng.shuffle(pool)
        take = min(quota[m], len(pool))
        out.extend(pool[:take])
    # final trim by shuffled order to n (keeps month spread)
    if len(out) > n:
        rng.shuffle(out)
        out = out[:n]
    return out


def build_question(prompt: str, evidence_fact: dict, keys: list[str]) -> dict:
    """Pair a paraphrased prompt with evidence fact + judgment keys."""
    return {
        "prompt": prompt,
        "evidence_id": evidence_fact.get("fact_id"),
        "keys": keys,
        "evidence_preview": (evidence_fact.get("content", "") or "")[:100],
    }


def extract_keys(content: str) -> list[str]:
    """Distinctive tokens for evidence-key judgment.

    EXCLUDES date stamps ([L|yymmdd] headers and bare 19xx/20xx years) —
    same-date facts would otherwise false-confirm any question from that
    day (2026-08-30 harness bug: corpus R@5 inflated by date-key hits).
    Keeps tickers, percentages, decimals, latin identifiers, CJK runs.
    """
    # date header of THIS fact (and any yymmdd lookalike) must not judge
    header = re.match(r"\[L\|(\d{6})", content or "")
    banned = set()
    if header:
        banned.add(header.group(1))
    keys = []
    for m in re.findall(r"\d+(?:\.\d+)?%?|[A-Za-z]{1,2}\d{4,6}(?:\.HK)?", content or ""):
        if re.fullmatch(r"\d{6}", m) and (m.startswith("26") or m.startswith("25") or m in banned):
            continue  # yymmdd date stamp
        if re.fullmatch(r"(19|20)\d{2}", m):
            continue  # bare year
        if len(m) >= 3 or "%" in m:
            keys.append(m)
    bp = set("我们的了是在和有对与及其这那你他它们中大小上下不也为就没时地过被把".split()) | {
        "结论", "投资", "公司", "业务", "我们", "当时", "关于", "说明", "显示"}
    for m in re.findall(r"[\u4e00-\u9fff]{2,6}", content or ""):
        if m not in bp and not any(m in k for k in keys):
            keys.append(m)
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:6]


def judge_hit(question: dict, retrieved: dict | None) -> bool:
    """Evidence-key judgment: any key is a space-normalized substring of the
    retrieved fact's normalized content. Benchmark's own artifacts excluded."""
    if not isinstance(retrieved, dict):
        return False
    content = retrieved.get("content", "") or ""
    if _BENCH_MARKER in content:
        return False
    norm = re.sub(r"\s+", "", content)
    for key in question.get("keys", []):
        if key and re.sub(r"\s+", "", str(key)) in norm:
            return True
    return False


def compute_metrics(ranks: list[int], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """ranks: 1-based rank of the evidence fact per question; 0 = miss."""
    m: dict = {"count": len(ranks)}
    for k in ks:
        m[f"R@{k}"] = sum(1 for r in ranks if 0 < r <= k) / len(ranks) if ranks else 0.0
    m["MRR"] = sum(1 / r for r in ranks if r > 0) / len(ranks) if ranks else 0.0
    return m


class BenchmarkHarness:
    """Runs named contenders on the same question set, evidence-key judged."""

    def __init__(self, contenders: "list[tuple[str, Callable]] | None" = None):
        self.contenders: list[tuple[str, object]] = list(contenders or [])
        self.questions: list[dict] = []

    def run_all(self, limit: int = 10) -> dict:
        results: dict = {}
        for name, search_fn in self.contenders:
            ranks: list[int] = []
            t0 = time.time()
            for q in self.questions:
                retrieved = search_fn(q, limit)
                rank = 0
                for i, f in enumerate(retrieved, 1):
                    if judge_hit(q, f):
                        rank = i
                        break
                ranks.append(rank)
            dt = (time.time() - t0) / max(1, len(self.questions))
            metrics = compute_metrics(ranks)
            metrics["latency_ms"] = round(dt * 1000, 1)
            results[name] = metrics
        return results
