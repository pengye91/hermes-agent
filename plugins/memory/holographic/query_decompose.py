"""Multi-hop query decomposition (glm-5.3-flash via Coding Plan endpoint).

Only queries the router classified as kind=="multihop" reach this module
(decomposition on fact-lookup queries measurably hurts: -5pp, ACL-SRW
2025). Design mirrors llm_rerank.py: same provider pattern, same failure
containment (any error → [original query]), same anti-fabrication stance
(sub-queries are validated strings, never trusted blindly).

Cost: one glm-5.3-flash call per multihop query, ~free on the Coding Plan
subscription; skipped entirely for single queries.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

_DECOMP_MODEL = os.environ.get("HOLO_DECOMP_MODEL", "glm-5.3-flash")
_DECOMP_TIMEOUT_S = 20.0
MAX_SUBS = 4          # hard cap — more sub-queries = more noise, not recall
_MAX_SUB_LEN = 200    # sub-queries longer than this are model rambling

_DECOMP_PROMPT = """你是查询分解器。把下面的多实体查询拆成2-{max_subs}个各自独立可检索的子查询,
每个子查询只覆盖一个主体/一个可检索事实。不要回答问题,只拆解查询。

用户查询: {query}

输出: 纯 JSON 字符串数组, 例如 ["子查询1", "子查询2"]。不要输出其他文字。"""


class _DecomposeClient:
    """Thin OpenAI-compatible client (same pattern as _RerankClient)."""

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

    def decompose(self, query: str, max_subs: int) -> str:
        """Return raw LLM text. Raises on failure."""
        prompt = _DECOMP_PROMPT.format(query=query, max_subs=max_subs)
        body = json.dumps({
            "model": _DECOMP_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "thinking": {"type": "disabled"},
            "max_tokens": 400,
        }, ensure_ascii=False).encode()
        req = urllib.request.Request(
            self._base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._api_key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=_DECOMP_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


def _parse_sub_queries(raw) -> list[str]:
    """Parse + validate the LLM's JSON string array. Garbage → []."""
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
    out: list[str] = []
    for x in arr:
        if not isinstance(x, str):
            continue
        x = x.strip()
        if len(x) < 3 or len(x) > _MAX_SUB_LEN:
            continue
        if x not in out:
            out.append(x)
    return out


def decompose_query(query: str, client, max_subs: int = MAX_SUBS) -> list[str]:
    """Split a multihop query into independent sub-queries.

    Failure contract (test-pinned): unavailable client, LLM error, or
    unparseable output → [original query] — the caller retrieves with the
    raw query exactly as today. Empty query → [].
    """
    if not query or not query.strip():
        return []
    if client is None or not getattr(client, "available", False):
        return [query]
    try:
        raw = client.decompose(query, max_subs)
        subs = _parse_sub_queries(raw)
        if not subs:
            return [query]
        return subs[:max_subs]
    except Exception as e:
        logger.debug("query decomposition failed (using original): %s", e)
        return [query]
