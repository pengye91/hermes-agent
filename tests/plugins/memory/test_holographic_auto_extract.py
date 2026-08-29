"""Regression tests for #57682 — holographic auto_extract harvested
context-compaction handoff summaries into fact_store, and ran even when
configured off.

2026-08-29 UPDATE: the English-regex harvester (_auto_extract_facts) has been
replaced by the LLM distillation hook (session_distill.py). These tests now
verify the SURVIVING contract of ``on_session_end``:

1. Gate: every false-y value of ``auto_extract`` (string enum "false" etc.)
   must keep the hook disabled — no subprocess, no facts. (Unchanged #57682.)
2. Enabled: the hook fires the distill subprocess exactly once, and facts are
   produced by the subprocess (not inline) — so the provider store stays
   untouched by the hook itself.
3. Failure containment: subprocess errors never propagate.

The old "summary prose not harvested" concern is now handled downstream by
the LLM distiller's semantic filtering in distill_llm.py (prompt rule 4:
discard 客套/反思/过渡), no longer by regex here.
"""

from unittest import mock

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    is_compaction_summary_message,
)
from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic import session_distill


def _make_provider(tmp_path, **config):
    base = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    base.update(config)
    provider = HolographicMemoryProvider(config=base)
    provider.initialize(session_id="test-session")
    return provider


def _fact_contents(provider):
    return [f["content"] for f in provider._store.list_facts(limit=100)]


def _user(content, **extra):
    msg = {"role": "user", "content": content}
    msg.update(extra)
    return msg


def _ok_run(*args, **kwargs):
    return mock.Mock(returncode=0, stdout="", stderr="")


DECISION_MSG = "we decided to use PostgreSQL for the persistence layer"
SUMMARY_MSG = (
    f"{SUMMARY_PREFIX}\n## Historical Task Snapshot\n"
    "The project uses a kanban board for all dispatch. "
    "We agreed to route reviews through the fan-in consumer."
)


# ---------------------------------------------------------------------------
# Defect 1 — string-boolean gate (unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("off_value", [False, "false", "False", "no", "off", "0", None, ""])
def test_auto_extract_off_values_disable_extraction(tmp_path, off_value):
    provider = _make_provider(tmp_path, auto_extract=off_value)
    with mock.patch.object(session_distill.subprocess, "run", side_effect=_ok_run) as run:
        provider.on_session_end([_user(DECISION_MSG)])
    assert run.call_count == 0
    assert _fact_contents(provider) == []
    provider.shutdown()


# ---------------------------------------------------------------------------
# Enabled path — fires the distill subprocess, nothing inline
# ---------------------------------------------------------------------------


def test_enabled_fires_distill_subprocess_once(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill.subprocess, "run", side_effect=_ok_run) as run:
        provider.on_session_end([_user(DECISION_MSG)])
    assert run.call_count == 1
    # Inline store untouched — facts come from the subprocess pipeline
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_subprocess_failure_never_propagates(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill.subprocess, "run",
                           side_effect=RuntimeError("boom")):
        provider.on_session_end([_user(DECISION_MSG)])  # must not raise
    provider.shutdown()


# ---------------------------------------------------------------------------
# is_compaction_summary_message — public helper contract (unchanged)
# ---------------------------------------------------------------------------


def test_helper_detects_prefix_metadata_and_merged_forms():
    assert is_compaction_summary_message(_user(SUMMARY_MSG))
    assert is_compaction_summary_message(
        _user("anything", **{COMPRESSED_SUMMARY_METADATA_KEY: True})
    )
    assert is_compaction_summary_message(
        _user(f"prior tail\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_MSG}")
    )
    assert not is_compaction_summary_message(_user(DECISION_MSG))
    assert not is_compaction_summary_message(_user(""))
