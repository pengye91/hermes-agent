"""Optional LLM listwise rerank for fact_store search (Fix #3, 2026-08-30).

The statistical rerank (fts/jaccard/hrr/vec weights) tops out at surface
signals; queries like "我们当时对X的结论" need judgment to separate the
CONCLUSION fact from same-topic evidence/history. This module adds an
explicit opt-in stage: rerank="llm" on fact_store search.

Design constraints (from the deployment host):
- glm-5.3-flash via the Coding Plan endpoint, thinking disabled — the same
  provider pattern as session_distill/CLS (subscription, ~zero marginal
  cost, ~¥0.002/listwise call if it ever falls back to pay-per-token).
- Explicit calls only — the per-turn prefetch path never reranks with an
  LLM (latency + prompt-noise budget).
- Failure containment: ANY error → results in the original statistical
  order. No raise, no retry loop (the next explicit call can retry).
- Anti-fabrication: the LLM may only pick from the candidate ids; unknown
  ids are dropped, content is never taken from the LLM output.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

_RERANK_MODEL = os.environ.get("HOLO_RERANK_MODEL", "glm-5.3-flash")
_RERANK_TIMEOUT_S = 20.0
_MAX_CANDIDATES = 12  # listwise input cap (latency/cost)

_RERANK_PROMPT = """你是记忆检索结果重排器。用户查询在下方。从候选事实中选出真正相关的条目并排序:
最直接回答查询意图的排最前(注意: 查询问"结论/决策"时优先排结论句, 问"数据/证据"时优先排证据句,
问"历史/演进"时按时间正序)。只输出相关条目, 无关的全部丢弃。

用户查询: {query}

候选事实:
{items}

输出: 纯 JSON 数组, 元素是 fact_id 数字, 按相关度降序, 例如 [17, 3, 42]。不要输出其他文字。"""


class _RerankClient:
    """Thin OpenAI-compatible client for the listwise rerank call."""

    def __init__(self):
        self.available = False
        self._api_key = ""
        self._base_url = ""
        try:
            env_path = os.path.expanduser("~/.hermes/.env")
            env = {}
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env[k.strip()] = v.strip()
            self._api_key = env.get("GLM_API_KEY", "")
            self._base_url = env.get("GLM_BASE_URL") or \
                "https://open.bigmodel.cn/api/coding/paas/v4"
            self.available = bool(self._api_key)
        except Exception:
            self.available = False

    def rerank(self, query: str, items: list[dict], k: int) -> list[int]:
        """Return picked fact_ids (may be shorter than k). Raises on failure."""
        listed = "\n".join(
            f"- [{f['fact_id']}] {str(f.get('content', ''))[:160]}"
            for f in items[:_MAX_CANDIDATES]
        )
        prompt = _RERANK_PROMPT.format(query=query, items=listed)
        body = json.dumps({
            "model": _RERANK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "thinking": {"type": "disabled"},
            "max_tokens": 300,
        }, ensure_ascii=False).encode()
        req = urllib.request.Request(
            self._base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._api_key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=_RERANK_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode())
        return _parse_picks(data["choices"][0]["message"]["content"])


def _parse_picks(raw) -> list[int]:
    """Parse the LLM's JSON id list. Garbage → []."""
    if not raw:
        return []
    s = str(raw).strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) >= 3 else s.lstrip("`")
        s = s.strip()
        if s.lower().startswith("json"):
            s = s[4:]
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j <= i:
        return []
    try:
        arr = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for x in arr:
        if isinstance(x, int) and 0 < x < 10**9:
            out.append(x)
    return out


def _apply_picks(candidates: list[dict], picks: list[int]) -> list[dict]:
    """Reorder candidates by the LLM's picks; unknown ids ignored."""
    by_id = {f.get("fact_id"): f for f in candidates}
    out = []
    for pos, fid in enumerate(picks, 1):
        f = by_id.get(fid)
        if f is None:
            continue
        g = dict(f)
        g["rerank_pos"] = pos
        g["llm_rerank"] = True
        out.append(g)
    return out


_BOILERPLATE = {"我们", "当时", "结论", "是什么", "什么", "怎么", "如何", "哪个", "哪些",
                "请", "帮忙", "一下", "告诉", "之前", "讨论", "说过", "聊过", "这个",
                "那个", "到底", "还是", "以及", "关于", "对于", "请问"}


def _boilerplate_strip(query: str) -> str:
    """Drop conversational boilerplate from a judgment query.

    '我们当时对百合花的结论是什么' → '百合花' — the stripped form drives a
    SECOND candidate pool; template-heavy queries otherwise bury the actual
    subject under hundreds of '我们当时对X的结论' boilerplate facts.

    Handles both space-separated and unseparated CJK: split on whitespace/
    punctuation first, then remove boilerplate SUBSTRINGS from the remaining
    tokens (unseparated Chinese keeps 对/的/了 as glue chars to drop too).
    """
    import re
    _GLUE = "对的对了是在和跟与及被把给从向朝着种个只条"
    tokens = re.split(r"[\s，。？！,.\u3000]+", query)
    kept = []
    for t in tokens:
        if not t:
            continue
        # substring-remove boilerplate words (longest first)
        for bp in sorted(_BOILERPLATE, key=len, reverse=True):
            t = t.replace(bp, " ")
        # split again after removal (e.g. '我们当时对百合花的结论' → ' 百合花 结论')
        for piece in re.split(r"[\s]+", t):
            piece = piece.strip(_GLUE)
            if len(piece) >= 2 and piece not in _BOILERPLATE:
                kept.append(piece)
    return " ".join(kept)


def rerank_search(search_fn, query: str, client, *, rerank: str = "off",
                  k: int = 5, limit: int = 10, overfetch: int = 24) -> list[dict]:
    """Wrap a statistical search with the optional LLM rerank.

    search_fn(query, limit) -> list[dict] must return the statistical
    ranking. rerank='off' (default) returns it unchanged. rerank='llm'
    builds an OVERFETCHED candidate pool from TWO sources — the raw query
    AND its boilerplate-stripped subject form ('我们当时对X的结论是什么'
    → 'X'; template tokens otherwise bury the subject under hundreds of
    same-shape facts) — merged and deduped, then asks the client to
    pick/order. ANY failure returns the statistical order untouched.
    """
    results = search_fn(query, limit)
    if rerank != "llm" or not results:
        return results
    if client is None or not getattr(client, "available", False):
        return results

    # widen the pool: raw query + subject-only query, merged
    pool = list(results)
    seen = {f.get("fact_id") for f in pool}
    stripped = _boilerplate_strip(query)
    if stripped and stripped != query:
        for f in search_fn(stripped, overfetch):
            if f.get("fact_id") not in seen:
                pool.append(f)
                seen.add(f.get("fact_id"))
    elif len(pool) < overfetch:
        pool = search_fn(query, max(overfetch, limit))

    try:
        picks = client.rerank(query, pool, k)
        if not picks:
            return results
        return _apply_picks(pool, picks)
    except Exception as e:
        logger.debug("LLM rerank failed (keeping statistical order): %s", e)
        return results
