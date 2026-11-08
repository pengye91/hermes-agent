"""Tests: the DEFAULT search path's vector leg strips boilerplate (cycle C).

Forensics (2026-08-30): smart_search's temporal path already retrieves
with _subject_form — vector_candidates receives the stripped form there.
But the DEFAULT FactRetriever.search embeds the RAW query; boilerplate
('我们当时对X的结论是什么') dilutes the embedding vector. Cycle C moves
the strip into search() so every caller benefits.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


class RecordingStore(MemoryStore):
    """Capture what the vector leg actually receives."""
    seen_queries: list[str] = []

    def vector_candidates(self, query, top=10, category=None):
        RecordingStore.seen_queries.append(query)
        return super().vector_candidates(query, top=top, category=category)


@pytest.fixture
def recording_retriever(tmp_path):
    store = RecordingStore(str(tmp_path / "v.db"))
    store.add_fact("百合花定增结论: 不参与", category="memory")
    store.add_fact("港股AI双雄代码: 智谱=2513.HK", category="memory")
    ret = FactRetriever(store=store)
    yield ret
    store.close()
    RecordingStore.seen_queries.clear()


class TestVectorLegStrips:
    def test_boilerplate_removed_before_embedding(self, recording_retriever):
        recording_retriever.search("我们当时对百合花的结论是什么", limit=5)
        assert RecordingStore.seen_queries, "vector leg must have been called"
        # every embedded query must be boilerplate-free
        for q in RecordingStore.seen_queries:
            assert "我们当时" not in q, q
            assert "是什么" not in q, q

    def test_plain_subject_query_unchanged(self, recording_retriever):
        recording_retriever.search("百合花 定增", limit=5)
        assert any("百合花" in q for q in RecordingStore.seen_queries)

    def test_strip_never_empties_query(self, recording_retriever):
        # all-boilerplate query must still embed SOMETHING (fallback to raw)
        recording_retriever.search("是什么", limit=5)
        assert all(q for q in RecordingStore.seen_queries), \
            RecordingStore.seen_queries
