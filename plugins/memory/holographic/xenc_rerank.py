"""Cross-encoder rerank stage: bge-reranker-v2-m3 via ONNX Runtime.

Cycle 5 of the retrieval upgrade (2026-08-30 research brief, rec #2):
the weighted statistical rerank (fts/jaccard/hrr) is noise-prone at
production scale — the HRR bank SNR is 0.15 and the hand-tuned weights
measurably UNDERPERFORM plain hybrid (61% vs 68% R@5 on the 08-30
benchmark). A cross-encoder scores query–doc pairs jointly; deployed as
a single int8 ONNX file it needs no torch/transformers stack
(onnxruntime already installed) and no external service.

Failure contract (test-pinned): missing model, missing onnxruntime, or
ANY scoring error → statistical order passthrough. Never raises.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = os.path.expanduser(
    "~/.hermes/models/bge-reranker-v2-m3-onnx")
_DEFAULT_BATCH = 16
_MAX_DOC_CHARS = 256   # measured 2026-08-30: 512-char truncation → 8.2s/q
                      # on this 4-core box; the rerank signal lives in the
                      # head of a fact sentence, 256 halves latency at no
                      # measurable precision cost (smoke: same ordering)


class CrossEncoderReranker:
    """Lazy-loading ONNX cross-encoder scorer.

    availability = model file + tokenizer + onnxruntime, all present.
    First ``score`` call pays the session+tokenizer init (~1-2s).
    """

    def __init__(self, model_dir: str | None = None, batch: int = _DEFAULT_BATCH):
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._batch = batch
        self._session = None
        self._tokenizer = None
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        if not os.path.isfile(os.path.join(self._model_dir, "model_int8.onnx")):
            return False
        if not os.path.isfile(os.path.join(self._model_dir, "tokenizer.json")):
            return False
        return True

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(
            os.path.join(self._model_dir, "tokenizer.json"))
        # variable-length facts need padding to a rectangular batch —
        # caught by the 2026-08-30 smoke test (inhomogeneous shapes)
        self._tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
        self._session = ort.InferenceSession(
            os.path.join(self._model_dir, "model_int8.onnx"),
            providers=["CPUExecutionProvider"])

    def _score_pairs(self, query: str, docs: list[str]) -> np.ndarray:
        """Score query-doc pairs in one forward batch. Raises on failure."""
        self._ensure_loaded()
        texts = [d[:_MAX_DOC_CHARS] for d in docs]
        enc = self._tokenizer.encode_batch(
            [f"{query}</s>{t}" for t in texts])
        input_ids = np.array([e.ids for e in enc], dtype="int64")
        attention = np.array([e.attention_mask for e in enc], dtype="int64")
        outs = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention})
        logits = outs[0].reshape(-1)
        return np.asarray(logits, dtype="float32")

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Batched scoring with the internal batch size."""
        out: list[float] = []
        for i in range(0, len(docs), self._batch):
            out.extend(self._score_pairs(query, docs[i:i + self._batch]).tolist())
        return np.array(out, dtype="float32")


def rerank_candidates(query: str, candidates: list[dict],
                      reranker: CrossEncoderReranker | None) -> list[dict]:
    """Reorder candidates by cross-encoder relevance to the query.

    Passthrough (same order, zero risk) when the reranker is None /
    unavailable, the pool is empty, or scoring raises. On success each
    result dict gains ``xenc_score``.
    """
    if not candidates:
        return candidates
    if reranker is None or not getattr(reranker, "available", False):
        return candidates
    try:
        docs = [str(f.get("content", "")) for f in candidates]
        scores = reranker.score(query, docs)
        order = np.argsort(-scores)
        out = []
        for idx in order:
            g = dict(candidates[int(idx)])
            g["xenc_score"] = float(scores[idx])
            out.append(g)
        return out
    except Exception as e:
        logger.debug("cross-encoder rerank failed (passthrough): %s", e)
        return candidates
