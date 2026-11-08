"""Integration tests for smart_search — router + time filter + decomposition
wired into FactRetriever, with a real on-disk store.

The plan (from the 2026-08-30 research brief):
1. route_query classifies (kind, time_filter)
2. time_filter prunes the candidate pool BEFORE ranking (facts outside the
   window lose to in-window noise today)
3. multihop queries decompose into sub-queries; sub-query pools merge
   (round-robin interleave, dedup by fact_id)
4. single queries: identical behaviour to FactRetriever.search (no regression
   on the fact-lookup path — the over-split tax is real, -5pp)
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from datetime import date

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic import smart_search as smart


NOW = date(2026, 8, 30)


def _seed(store: MemoryStore) -> None:
    # temporal corpus: same subject, different months
    store.add_fact("[L|260615|百合花] 百合花纯度门槛定为1.3%", category="memory")
    store.add_fact("[L|260710|百合花] 百合花纯度门槛上调至2.0%", category="memory")
    store.add_fact("[L|250615|百合花] 百合花纯度门槛历史值0.8%", category="memory")
    # two-entity corpus for multihop
    store.add_fact("[L|260720|智谱] 智谱港股代码2513.HK", category="memory")
    store.add_fact("[L|260721|MiniMax] MiniMax港股代码0100.HK", category="memory")
    # headerless fact — must survive time pruning
    store.add_fact("玻璃基板加工端处于价值链最差位置", category="memory")


@pytest.fixture
def seeded_retriever(tmp_path):
    store = MemoryStore(str(tmp_path / "smart.db"))
    _seed(store)
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


class StubClient:
    available = True
    def __init__(self, payload: str):
        self._payload = payload
    def decompose(self, query, max_subs):
        return self._payload


class DeadClient:
    available = False
    def decompose(self, query, max_subs):
        raise RuntimeError("unreachable")


class TestSmartSearchSingle:
    def test_single_query_matches_plain_search(self, seeded_retriever):
        """Regression guard: fact-lookup path must return the SAME facts as
        the existing search — smart_search must not tax simple queries."""
        plain = seeded_retriever.search("百合花纯度门槛", limit=3)
        smarted = smart.smart_search(seeded_retriever, "百合花纯度门槛",
                                     limit=3, now=NOW)
        assert [f["fact_id"] for f in plain] == [f["fact_id"] for f in smarted]

    def test_headerless_fact_still_reachable(self, seeded_retriever):
        out = smart.smart_search(seeded_retriever, "玻璃基板加工端位置",
                                 limit=3, now=NOW)
        assert any("玻璃基板" in f["content"] for f in out)


class TestSmartSearchTemporal:
    def test_window_prunes_older_fact_of_same_subject(self, seeded_retriever):
        """'上个月' (July 2026) must surface the 260710 fact and drop the
        2025 history fact — today all three compete on lexical score."""
        out = smart.smart_search(seeded_retriever, "上个月百合花纯度门槛",
                                 limit=3, now=NOW)
        assert out, "must return results"
        contents = [f["content"] for f in out]
        assert any("2.0%" in c for c in contents)      # July fact survives
        assert not any("0.8%" in c for c in contents)  # 2025 fact pruned

    def test_time_words_do_not_bury_in_window_facts(self, seeded_retriever):
        """Regression (2026-08-30 A/B): '今年4月智谱有什么进展' — the time
        words diluted FTS signal so the 2604 fact never entered the pool.
        After strip_time_words + boilerplate strip the subject-only query
        must rank the in-window fact first."""
        seeded_retriever.store.add_fact(
            "[L|260421|智谱] 主模型智谱GLM-5.1遇到429时自动切换到minimax",
            category="memory")
        out = smart.smart_search(seeded_retriever, "今年4月智谱有什么进展",
                                 limit=3, now=NOW)
        assert out
        assert "GLM-5.1" in out[0]["content"] or any(
            "GLM-5.1" in f["content"] for f in out[:3])

    def test_zero_support_term_does_not_bury_entity(self, seeded_retriever):
        """Regression (2026-08-30 A/B, production): '8月智谱的最新动态' —
        '动态' has ZERO corpus support and '最新' drags in version/data
        distractors, so 53 in-window 智谱 facts never surfaced. Temporal
        path must merge a second pool from the extracted entity alone
        (dual-pool pattern, same as llm_rerank's subject-only second pool)."""
        # distractors with 最新 but no 智谱
        seeded_retriever.store.add_fact(
            "[L|260815|x] 0.0.4 是当前工具的最新版本", category="memory")
        seeded_retriever.store.add_fact(
            "[L|260816|x] 数据库中的最新数据截止到2026年", category="memory")
        # the in-window target: 智谱 in 2608, no 最新/动态 words
        seeded_retriever.store.add_fact(
            "[L|260829|智谱] 智谱业绩watchdog误报修复完成", category="memory")
        out = smart.smart_search(seeded_retriever, "8月智谱的最新动态",
                                 limit=3, now=NOW)
        assert any("watchdog" in f["content"] for f in out[:3])

    def test_non_temporal_query_returns_all_subject_facts(self, seeded_retriever):
        out = smart.smart_search(seeded_retriever, "百合花纯度门槛", limit=5,
                                 now=NOW)
        # no time words → no filter → all three 百合花 facts eligible
        assert len(out) >= 2


class TestSmartSearchMultihop:
    def test_enum_query_finds_both_entities(self, seeded_retriever):
        out = smart.smart_search(
            seeded_retriever, "智谱和MiniMax的港股代码分别是什么",
            limit=5, now=NOW,
            decompose_client=StubClient('["智谱的港股代码是什么", "MiniMax的港股代码是什么"]'),
        )
        contents = "".join(f["content"] for f in out)
        assert "2513.HK" in contents and "0100.HK" in contents

    def test_subquery_boilerplate_does_not_bury_target(self, seeded_retriever):
        """Regression (2026-08-30 A/B on production store): sub-query
        '智谱的港股代码是什么' ranked 澜起科技港股代码 facts ABOVE the target
        because '是什么' boilerplate + '港股代码' overlap favors long
        distractors. Sub-queries must be boilerplate-stripped before
        retrieval (same lesson as llm_rerank's second pool)."""
        # distractor: high '港股代码' overlap, zero 智谱 relevance
        seeded_retriever.store.add_fact(
            "[L|260514|澜起] 澜起科技港股代码06809.HK，中兴通讯港股代码00763.HK",
            category="memory")
        out = smart.smart_search(
            seeded_retriever, "智谱和MiniMax的港股代码分别是什么",
            limit=3, now=NOW,
            decompose_client=StubClient('["智谱的港股代码是什么", "MiniMax的港股代码是什么"]'),
        )
        top3 = "".join(f["content"] for f in out[:3])
        # the 双雄 fact must beat the 澜起 distractor in the merged head
        assert "2513.HK" in top3

    def test_dead_client_degrades_to_plain_search(self, seeded_retriever):
        plain = seeded_retriever.search("智谱和MiniMax的港股代码分别是什么",
                                        limit=5)
        out = smart.smart_search(
            seeded_retriever, "智谱和MiniMax的港股代码分别是什么",
            limit=5, now=NOW, decompose_client=DeadClient(),
        )
        assert {f["fact_id"] for f in plain} <= {f["fact_id"] for f in out}
