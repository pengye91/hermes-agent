"""Optional embedding leg for the holographic memory store.

Zero hard dependencies (stdlib urllib only). When no API key is configured
the module reports unavailable and the store degrades to pure lexical
retrieval — exactly the pre-embedding behaviour.

Default provider targets the OpenAI-compatible /embeddings surface, which
GLM (bigmodel.cn), OpenAI, DeepSeek and most gateways implement. The client
mirrors the production-proven pattern of the CLS news pipeline: small
512-dim vectors, batch calls, BLOB storage, brute-force cosine in numpy.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Env vars probed in order when no explicit key is configured.
_KEY_ENV_VARS = (
    "HOLO_EMBEDDING_API_KEY",
    "GLM_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
)


class EmbeddingClient:
    """Thin OpenAI-compatible embeddings client with availability probe."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int = 512,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 15.0,
    ):
        self.base_url = (base_url or "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
        self.model = model or "embedding-3"
        self.dimensions = int(dimensions)
        self.timeout = timeout

        if api_key_env:
            candidates = (api_key_env,)
        else:
            candidates = _KEY_ENV_VARS
        self.api_key = api_key or next(
            (os.environ.get(v) for v in candidates if os.environ.get(v)), None
        )

    # -- availability ------------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"EmbeddingClient(model={self.model}, dim={self.dimensions}, ok={self.available})"

    # -- API ---------------------------------------------------------------

    def embed(self, texts: list[str], retries: int = 2) -> list[list[float]] | None:
        """Embed a batch of texts. Returns None on failure (never raises)."""
        if not self.available or not texts:
            return None
        body = json.dumps(
            {
                "model": self.model,
                "input": [t[:4000] for t in texts],
                "dimensions": self.dimensions,
            }
        ).encode()
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/embeddings",
                    data=body,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                out = [d["embedding"] for d in data["data"]]
                if len(out) == len(texts):
                    return out
                last_err = ValueError(
                    f"embedding count mismatch: {len(out)} != {len(texts)}"
                )
            except Exception as e:  # noqa: BLE001 - degrade, never raise
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        logger.debug("embedding call failed: %s", last_err)
        return None


def pack_vector(vec: list[float]) -> bytes:
    """Serialize float32 vector to BLOB (half the size of float64)."""
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    import struct

    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine(a, b) -> float:
    """Cosine similarity; accepts sequences (lists or numpy arrays)."""
    try:
        import numpy as np

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except ImportError:  # pragma: no cover - numpy optional
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)


def load_env_file(path: str | Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE lines) for explicit-path configs."""
    env: dict[str, str] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except OSError:
        pass
    return env
