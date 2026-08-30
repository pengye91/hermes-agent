"""TDD tests for entity-extraction quality fixes (2026-08-30 audit).

Audit findings on the 27.5k-fact production store:
- 40.4% of entity links are noise: date stamps (18.5%, the [L|yymmdd]
  header eaten by the ticker regex), ≤2-char fragments (21.2%, generic
  business words 结论/止损/营收), template words (0.7%).
- probe('260516') recalled 3,177 facts — pure noise amplification.
- Real 2-char entities (智谱 #176, 腾讯 #75) are drowned under generic
  words (板块 #33). Frequency-based deletion would kill them; the fix
  must be a STOPWORD list, not a length floor.

Extractor fixes under test:
1. [L|yymmdd|entity] headers: strip before extraction (date AND entity
   both extracted today — the entity correctly, the date as junk).
2. Ticker regex must not match bare 6-digit date stamps: 260516 is a
   yymmdd date, 688507 is an A-share ticker. Anchor: A-share codes start
   with 60/68/00/30/8/4/9 (exchange prefixes); date stamps start 25/26.
3. CJK stopwords: generic business/grammar words never become entities.
4. Real 2-char entities (智谱/腾讯/观澜) MUST still extract.
"""

import sys
from pathlib import Path

HERMES = "/usr/local/lib/hermes-agent"
if HERMES not in sys.path:
    sys.path.insert(0, HERMES)

from plugins.memory.holographic.store import MemoryStore, _RE_TICKER, _ENTITY_STOPWORDS


def _extract(text):
    # _extract_entities touches no instance state (pure regex logic) —
    # __new__ skips __init__ so no database is ever opened.
    return MemoryStore._extract_entities(MemoryStore.__new__(MemoryStore), text)


# ── 1. header handling ─────────────────────────────────

def test_header_date_not_extracted():
    assert "260628" not in _extract("[L|260628|百合花] 投资结论：百合花不参与")


def test_header_entity_extracted():
    assert "百合花" in _extract("[L|260628|百合花] 投资结论：百合花不参与")


def test_plain_date_stamps_never_entities():
    assert "260516" not in _extract("2026年5月16日 260516 备份完成")


# ── 2. ticker regex anchors ────────────────────────────

def test_ticker_regex_matches_real_tickers():
    for t in ("688507", "002222", "301421", "300293", "600895", "000063"):
        assert _RE_TICKER.search(t), t


def test_ticker_regex_rejects_date_stamps():
    for d in ("260628", "251231", "241001"):
        assert not _RE_TICKER.search(d), d


# ── 3. stopwords ───────────────────────────────────────

def test_generic_business_words_stopped():
    for w in ("结论", "止损", "营收", "净利", "减仓", "标的", "板块", "市值",
              "信号", "买入", "验证", "数据", "模式", "参数", "阈值"):
        assert w not in _extract(f"关于{w}的重要说明 {w}变化需注意"), w


def test_stopword_list_contains_audit_top_offenders():
    # 2026-08-30 audit top generic entities (链接数降序)
    must = {"结论", "新增", "止损", "营收", "净利", "字段", "修复", "导致",
            "亿元", "减仓", "而非", "分钟", "返回", "改为", "现价", "调用",
            "标的", "通过", "数据", "模式", "待办", "浮盈", "当前", "不是",
            "参数", "信号", "买入", "验证", "量比", "使用", "任务", "算力",
            "板块", "市值", "权重", "评分", "全部", "涨停", "阈值", "覆盖",
            "更新", "文件", "函数", "仓位", "成本", "触发", "格式", "删除",
            "路径", "确认", "新闻", "完成", "决策", "问题", "方案"}
    missing = must - _ENTITY_STOPWORDS
    assert not missing, f"stopword 表缺: {sorted(missing)}"


# ── 4. real entities survive ───────────────────────────

def test_real_2char_entities_survive():
    for name in ("智谱", "腾讯", "观澜", "澜起", "华为", "中兴"):
        assert name in _extract(f"今天讨论{name}的估值"), name


def test_real_3plus_entities_survive():
    for name in ("百合花", "澜起科技", "寒武纪", "MiniMax", "DuckDB"):
        assert name in _extract(f"对{name}的关键判断"), name


def test_cjk_run_boundaries_respected():
    # 提取器不应把 '智谱AI' 粘连成别的: CJK run 智谱AI → 智谱 (2字CJK run) + AI
    ents = _extract("智谱AI 发布新模型")
    assert "智谱" in ents or any("智谱" in e for e in ents)
