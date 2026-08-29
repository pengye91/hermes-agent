"""TDD tests for the session-end LLM distillation hook (holographic).

Contract:
1. ``on_session_end`` with auto_extract on triggers the distill subprocess
   exactly once, with the venv python, and never raises.
2. auto_extract off (default) → no subprocess at all.
3. Hook failure (non-zero rc / timeout / missing script) is swallowed —
   the memory manager's worker must not die.
4. Empty messages → no subprocess.
"""

import os
import sys
from unittest import mock

import pytest

HERMES_ROOT = "/usr/local/lib/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

from plugins.memory.holographic import session_distill  # noqa: E402
from plugins.memory.holographic import HolographicMemoryProvider  # noqa: E402


def _make_provider(tmp_path, **config):
    base = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    base.update(config)
    provider = HolographicMemoryProvider(config=base)
    provider.initialize(session_id="test-session")
    return provider


def _ok_run(*args, **kwargs):
    return mock.Mock(returncode=0, stdout="", stderr="")


# ── gate ─────────────────────────────────────────────────

def test_session_end_triggers_distill(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill.subprocess, "run", side_effect=_ok_run) as run:
        provider.on_session_end([{"role": "user", "content": "我们讨论了百合花，结论不参与"}])
    assert run.call_count == 1
    cmd = run.call_args[0][0]
    assert cmd[0].endswith("python3") or "venv" in cmd[0]
    assert cmd[1].endswith("distill_llm.py")
    assert "distill" in cmd
    provider.shutdown()


def test_session_end_off_by_default(tmp_path):
    provider = _make_provider(tmp_path)  # no auto_extract
    with mock.patch.object(session_distill.subprocess, "run", side_effect=_ok_run) as run:
        provider.on_session_end([{"role": "user", "content": "x" * 50}])
    assert run.call_count == 0
    provider.shutdown()


# ── failure containment ─────────────────────────────────

def test_hook_never_raises_on_subprocess_failure(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill.subprocess, "run",
                           side_effect=RuntimeError("boom")):
        provider.on_session_end([{"role": "user", "content": "y" * 50}])  # no raise
    provider.shutdown()


def test_hook_swallows_nonzero_rc(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill.subprocess, "run",
                           return_value=mock.Mock(returncode=2, stdout="", stderr="quota")):
        rc = provider.on_session_end([{"role": "user", "content": "z" * 50}])
    assert rc is None or rc != "crash"  # returned without raising
    provider.shutdown()


def test_hook_timeout_contained(tmp_path):
    import subprocess as sp
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill.subprocess, "run",
                           side_effect=sp.TimeoutExpired(cmd="x", timeout=1)):
        provider.on_session_end([{"role": "user", "content": "t" * 50}])  # no raise
    provider.shutdown()


def test_empty_messages_no_subprocess(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill.subprocess, "run", side_effect=_ok_run) as run:
        provider.on_session_end([])
    assert run.call_count == 0
    provider.shutdown()


def test_missing_script_skips(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill, "_DISTILL_SCRIPT", "/nonexistent/distill.py"):
        with mock.patch.object(session_distill.subprocess, "run", side_effect=_ok_run) as run:
            provider.on_session_end([{"role": "user", "content": "m" * 50}])
    assert run.call_count == 0
    provider.shutdown()


# ── runner unit ─────────────────────────────────────────

def test_run_session_distillation_returns_rc():
    with mock.patch.object(session_distill.subprocess, "run", side_effect=_ok_run):
        assert session_distill.run_session_distillation([{"role": "user", "content": "x"}]) == 0
