"""Query-type router for FactRetriever retrieval strategy selection.

Classifies a user query into (kind, time_filter) before hybrid retrieval:

- kind: "single" (fact-lookup — retrieve with the raw query; decomposition
  measurably HURTS here, -5pp, ACL-SRW 2025) or "multihop" (decompose into
  sub-queries; MRR@10 +36.7% on multi-hop questions).
- time_filter: optional (start, end) window in yymmdd format matching the
  ``[L|yymmdd]`` header convention of distilled facts. LongMemEval reports
  +11.3% recall from time-aware pruning — our facts carry the timestamp in
  content, so the filter is a cheap post-retrieval constraint.

Design constraints (host policy):
- PURE function: no LLM, no DB, no network. Free and deterministic.
- Chinese-first: markers are CJK n-grams; English queries fall through to
  "single" harmlessly (English path unchanged).
- Conservative multihop trigger: only explicit enumeration/comparison
  markers route to decomposition. Over-triggering is the documented failure
  mode (the "simple-query tax").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import re


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

# Explicit multi-entity signals. "和" alone is NOT enough ("玻璃基板和设备"
# is one topic); require an enumeration/comparison marker nearby.
_MULTIHOP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"分别"),
    re.compile(r"哪个更|哪一个更|谁更|谁涨幅|谁的"),
    re.compile(r"对比|比较一下|区别|差异"),
    re.compile(r"vs\.?", re.IGNORECASE),
    re.compile(r"A股和港[股A]"),  # cross-market comparison shape
)

_ABS_YEAR_MONTH = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月")
_ABS_YEAR = re.compile(r"(20\d{2})\s*年")
_BARE_MONTH = re.compile(r"(?<![0-9\d])([1-9]|1[0-2])\s*月(?!份)")
_THIS_YEAR = re.compile(r"今年")
_LAST_YEAR = re.compile(r"去年|上一年")
_FIRST_HALF = re.compile(r"上半年")
_SECOND_HALF = re.compile(r"下半年")
_LAST_MONTH = re.compile(r"上个月|上月")
_THIS_MONTH = re.compile(r"本月|这个月")
_RECENT_DAYS = re.compile(r"最近(\d{1,3})天|近(\d{1,2})天")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _month_end(d: date) -> date:
    nxt = d.replace(day=28) + timedelta(days=4)
    return nxt.replace(day=1) - timedelta(days=1)


def _prev_month(n: date) -> tuple[date, date]:
    first = n.replace(day=1)
    last_prev = first - timedelta(days=1)
    return last_prev.replace(day=1), last_prev


def _fmt(d: date) -> str:
    return f"{d.year % 100:02d}{d.month:02d}{d.day:02d}"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryPlan:
    query: str
    kind: str = "single"                        # "single" | "multihop"
    time_filter: tuple[str, str] | None = None  # (yymmdd_start, yymmdd_end)
    sub_queries: list[str] = field(default_factory=list)

    @property
    def is_temporal(self) -> bool:
        return self.time_filter is not None


def route_query(query: str, now: date | None = None) -> QueryPlan:
    """Classify ``query`` into a retrieval strategy + optional time window.

    Always returns a plan (never raises); unknown/empty queries degrade to
    the plain single-path — identical to today's behaviour.
    """
    if now is None:
        now = date.today()

    q = (query or "").strip()
    if not q:
        return QueryPlan(query=query or "")

    kind = "multihop" if any(p.search(q) for p in _MULTIHOP_PATTERNS) else "single"
    time_filter = _resolve_time_window(q, now)

    return QueryPlan(query=query, kind=kind, time_filter=time_filter)


def _resolve_time_window(q: str, now: date) -> tuple[str, str] | None:
    # 1. Absolute year+month — highest precedence, unambiguous.
    m = _ABS_YEAR_MONTH.search(q)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            start = date(y, mo, 1)
            return _fmt(start), _fmt(_month_end(start))

    # 2. Explicit recent-N-days ("最近14天" / "近7天").
    m = _RECENT_DAYS.search(q)
    if m:
        days = int(m.group(1) or m.group(2))
        try:
            start = now - timedelta(days=days - 1)
        except (ValueError, OverflowError):
            return None
        return _fmt(start), _fmt(now)

    # 3. Absolute year alone ("2025年…" but NOT part of "2025年6月", already
    #    consumed by rule 1).
    m = _ABS_YEAR.search(q)
    if m:
        y = int(m.group(1))
        if y <= now.year:
            return _fmt(date(y, 1, 1)), _fmt(date(y, 12, 31))
        return None  # future year — no sane window, treat as non-temporal

    # 4. Relative words (checked longest-first to keep 上下半年/去今年 pairs
    #    unambiguous).
    if _LAST_YEAR.search(q):
        y = now.year - 1
        return _fmt(date(y, 1, 1)), _fmt(date(y, 12, 31))
    if _THIS_YEAR.search(q):
        return _fmt(date(now.year, 1, 1)), _fmt(now)
    if _FIRST_HALF.search(q):
        return _fmt(date(now.year, 1, 1)), _fmt(date(now.year, 6, 30))
    if _SECOND_HALF.search(q):
        return _fmt(date(now.year, 7, 1)), _fmt(date(now.year, 12, 31))
    if _LAST_MONTH.search(q):
        s, e = _prev_month(now)
        return _fmt(s), _fmt(e)
    if _THIS_MONTH.search(q):
        return _fmt(now.replace(day=1)), _fmt(_month_end(now))

    # 5. Bare month ("3月", "12月") → most recent PAST month with that number.
    m = _BARE_MONTH.search(q)
    if m:
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            year = now.year if mo <= now.month else now.year - 1
            start = date(year, mo, 1)
            return _fmt(start), _fmt(_month_end(start))

    return None


# ---------------------------------------------------------------------------
# Time-word stripping (temporal queries retrieve better without them)
# ---------------------------------------------------------------------------

# Ordered longest-first so "2026年6月" dies before "2026年" and "6月".
_TIME_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"20\d{2}\s*年\s*\d{1,2}\s*月(份)?"),
    re.compile(r"(今|去|上?)年\s*\d{1,2}\s*月(份)?"),
    re.compile(r"(?:最近|近)\s*\d{1,3}\s*天"),
    re.compile(r"20\d{2}\s*年"),
    re.compile(r"(今年|去年|上一年|上半年|下半年|上个月|上月|本月|这个月)"),
    re.compile(r"(?<![0-9\d])([1-9]|1[0-2])\s*月(?!份)"),
)


def strip_time_words(query: str | None) -> str:
    """Remove temporal phrases from a query; the window filter carries the
    temporal intent. Time words in the retrieval query are pure noise —
    they dilute FTS/vector signal and bury in-window facts (2026-08-30 A/B
    finding: 2604 智谱 facts never entered the top-30 pool).
    """
    if not query:
        return ""
    out = query
    for pat in _TIME_PHRASES:
        out = pat.sub(" ", out)
    # collapse the whitespace left behind
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Time-window application over candidate pools
# ---------------------------------------------------------------------------

# ``[L|yymmdd|主体]`` headers, e.g. [L|260615|百合花] — 6-digit date is
# always the second field. Also tolerates a header missing the subject
# ([L|260615]) — the distiller writes both shapes historically.
_HEADER_DATE = re.compile(r"\[L\|(\d{6})")


def apply_time_filter(
    facts: list[dict], time_filter: tuple[str, str] | None
) -> list[dict]:
    """Keep facts whose ANY ``[L|yymmdd]`` header date falls in the window.

    Policy (test-pinned):
    - Header-less facts are NEVER pruned (unknown date ≠ outside window —
      curated imports and migrated stores carry no headers).
    - If pruning would empty the pool, return the pool unchanged — graceful
      degradation beats empty recall.
    - ``time_filter is None`` → passthrough (same object, zero work).
    """
    if time_filter is None:
        return facts

    start, end = time_filter
    kept: list[dict] = []
    for f in facts:
        dates = _HEADER_DATE.findall(f.get("content", "") or "")
        if not dates:               # header-less → always keep
            kept.append(f)
        elif any(start <= d <= end for d in dates):
            kept.append(f)          # any header in window → keep
        # else: all headers outside window → pruned

    return kept if kept else facts
