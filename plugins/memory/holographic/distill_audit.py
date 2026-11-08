"""Distillation qualifier-retention audit (cycle 8).

Verifies — on real data — that the LLM distiller actually keeps the
qualifiers its prompt demands (rule 2: action verbs; rule 3: numbers and
dates). The failure class it hunts is the CogCanvas one: summarization
keeps the topic but silently drops 不/全部/only/数字, turning "不参与"
into "参与" downstream.

Read-only: joins facts (msg_id provenance column) back to their source
messages in hermes state.db and reports per-class retention + worst
examples. Never writes, never raises past argument validation.
"""
from __future__ import annotations

import re
import sqlite3

# --- qualifier classes -----------------------------------------------------

NEG_PAT = re.compile(r"不参与|不买入|不建议|不行|没有|无|未|勿|别|不再|放弃")
DATE_PAT = re.compile(
    r"\d{4}\s*年|\d{1,2}\s*月\d{1,2}\s*日|\d{1,2}\s*月|"
    r"\d+(\.\d+)?\s*%|\.?\d{4,6}|[A-Za-z]?\d{2,5}\.(HK|SZ|SH)")
QUANT_PAT = re.compile(r"全部|只|均|每|最多|至少|一律|统统|仅|半仓|满仓|清仓")


def extract_qualifiers(text: str) -> dict[str, list[str]]:
    return {
        "dates": DATE_PAT.findall(text or ""),
        "negatives": NEG_PAT.findall(text or ""),
        "quantifiers": QUANT_PAT.findall(text or ""),
    }


def audit_pair(source: str, fact: str) -> dict[str, float]:
    """Retention of each qualifier class: |kept| / |source|, vacuous 1.0.

    'Kept' = ANY member of the class present in the fact (class-level
    retention; per-token matching is too brittle against legitimate
    paraphrase, e.g. '不参与' → '放弃').
    """
    src_q = extract_qualifiers(source)
    fac_q = extract_qualifiers(fact)
    out = {}
    for cls in ("dates", "negatives", "quantifiers"):
        src_hits = src_q[cls]
        if not src_hits:
            out[f"{cls}_retained"] = 1.0
        else:
            kept = 1.0 if fac_q[cls] else 0.0
            out[f"{cls}_retained"] = kept
    return out


def sample_audit(db_path: str, limit: int = 200,
                 state_db_path: str = "~/.hermes/state.db") -> dict:
    """Sample distilled facts WITH msg_id, join to source messages, report.

    Returns aggregate retention per class + worst loss examples.
    Read-only on both DBs; missing tables/columns → empty report, no raise.
    """
    report = {"pairs": 0, "dates_retained": 0.0,
              "negatives_retained": 0.0, "quantifiers_retained": 0.0,
              "worst_examples": []}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        facts = conn.execute(
            "SELECT fact_id, content, msg_id FROM facts "
            "WHERE msg_id IS NOT NULL AND msg_id > 0 "
            "ORDER BY RANDOM() LIMIT ?", (limit,)).fetchall()
        conn.close()
    except sqlite3.Error:
        return report
    if not facts:
        return report

    import os
    state = os.path.expanduser(state_db_path)
    if not os.path.exists(state):
        return report
    try:
        sconn = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    except sqlite3.Error:
        return report

    sums = {"dates": 0.0, "negatives": 0.0, "quantifiers": 0.0}
    n = 0
    for fid, fcontent, msg_id in facts:
        try:
            row = sconn.execute(
                "SELECT content FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()
        except sqlite3.Error:
            continue
        if not row or not row[0]:
            continue
        src_text = row[0] if isinstance(row[0], str) else str(row[0])
        r = audit_pair(src_text, fcontent or "")
        n += 1
        for cls in sums:
            sums[cls] += r[f"{cls}_retained"]
            if r[f"{cls}_retained"] == 0.0 and len(report["worst_examples"]) < 20:
                report["worst_examples"].append(
                    {"fact_id": fid, "cls": cls,
                     "src": src_text[:120], "fact": (fcontent or "")[:120]})
    sconn.close()

    if n:
        report["pairs"] = n
        for cls in sums:
            report[f"{cls}_retained"] = sums[cls] / n
    return report
