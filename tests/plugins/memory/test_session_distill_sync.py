"""TDD tests for the sync_turn debounce distillation scheduler.

Contract (from the design discussion 2026-08-29):
1. sync_turn notes turn activity and schedules a distill after a 5min quiet
   window — on an INDEPENDENT thread; sync_turn itself returns immediately
   and never blocks the memory manager's serialized worker.
2. Active conversation (turns closer than the debounce) does not fire —
   until the 30min hard cap is reached, which fires even while active.
3. Only one distill subprocess in flight at a time (flock + _running guard).
4. auto_extract off → sync_turn is a no-op.
5. shutdown cancels pending timers.

Timing constants are module attrs so tests monkeypatch them to milliseconds.
"""

import os
import sys
import threading
import time
from unittest import mock

import pytest

HERMES_ROOT = "/usr/local/lib/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

from plugins.memory.holographic import session_distill  # noqa: E402
from plugins.memory.holographic import HolographicMemoryProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _fast_windows(monkeypatch):
    monkeypatch.setattr(session_distill, "_DEBOUNCE_S", 0.10)
    monkeypatch.setattr(session_distill, "_HARD_CAP_S", 0.40)
    session_distill.reset_scheduler()
    yield
    session_distill.reset_scheduler()


class _RunRecorder:
    """Thread-safe subprocess.run fake with optional blocking."""

    def __init__(self, block_s=0.0):
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
        self._block_s = block_s

    def __call__(self, *a, **kw):
        with self._lock:
            self.calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            time.sleep(self._block_s)
            return mock.Mock(returncode=0, stdout="", stderr="")
        finally:
            with self._lock:
                self.concurrent -= 1


def _wait_for(cond, timeout=5.0, step=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(step)
    return cond()


# ── 1. quiet window fires distill ───────────────────────

def test_quiet_window_fires_distill_once():
    rec = _RunRecorder()
    with mock.patch.object(session_distill.subprocess, "run", side_effect=rec):
        session_distill.note_turn()  # turn happens, then silence
        assert _wait_for(lambda: rec.calls >= 1), "distill should fire after quiet window"
        time.sleep(0.15)             # no re-fire without new turns
        assert rec.calls == 1


def test_sync_turn_returns_fast():
    """sync_turn path must return in milliseconds (never block the worker)."""
    rec = _RunRecorder(block_s=0.5)
    with mock.patch.object(session_distill.subprocess, "run", side_effect=rec):
        t0 = time.time()
        session_distill.note_turn()
        dt = time.time() - t0
        assert dt < 0.05, f"note_turn must be instant, took {dt:.3f}s"


# ── 2. active conversation: debounce + hard cap ─────────

def test_active_conversation_defers_until_quiet():
    rec = _RunRecorder()
    with mock.patch.object(session_distill.subprocess, "run", side_effect=rec):
        for _ in range(6):           # a turn every 60ms < debounce 100ms
            session_distill.note_turn()
            time.sleep(0.06)
        assert rec.calls == 0, "must not fire while conversation is active"
        assert _wait_for(lambda: rec.calls >= 1), "fires once it goes quiet"


def test_hard_cap_fires_during_continuous_activity():
    # debounce 100ms, cap 400ms: turns every 60ms keep it 'active',
    # but after 400ms the hard cap must fire anyway.
    rec = _RunRecorder()
    with mock.patch.object(session_distill.subprocess, "run", side_effect=rec):
        deadline = time.time() + 1.2
        fired_early = False
        while time.time() < deadline:
            session_distill.note_turn()
            time.sleep(0.06)
            if rec.calls > 0 and time.time() - deadline < -0.8:
                fired_early = True  # fired well before the cap window
                break
        assert rec.calls >= 1, "hard cap must fire during continuous activity"
        assert not fired_early, "must not fire before the hard cap while active"


# ── 3. single-flight ────────────────────────────────────

def test_no_concurrent_distills():
    rec = _RunRecorder(block_s=0.3)
    with mock.patch.object(session_distill.subprocess, "run", side_effect=rec):
        session_distill.note_turn()
        assert _wait_for(lambda: rec.calls >= 1)
        session_distill.note_turn()  # second trigger while first still running
        session_distill.note_turn()
        assert _wait_for(lambda: rec.max_concurrent >= 1)
        time.sleep(0.5)
    assert rec.max_concurrent == 1, "subprocess must be single-flight"


# ── 4. provider gate + integration ──────────────────────

def _make_provider(tmp_path, **config):
    base = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    base.update(config)
    provider = HolographicMemoryProvider(config=base)
    provider.initialize(session_id="test-session")
    return provider


def test_provider_sync_turn_gated_off_by_default(tmp_path):
    provider = _make_provider(tmp_path)
    with mock.patch.object(session_distill, "note_turn") as nt:
        provider.sync_turn("user says", "assistant replies", session_id="s")
    assert nt.call_count == 0
    provider.shutdown()


def test_provider_sync_turn_notes_when_enabled(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill, "note_turn") as nt:
        provider.sync_turn("user says", "assistant replies", session_id="s")
    assert nt.call_count == 1
    provider.shutdown()


def test_provider_sync_turn_never_raises(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    with mock.patch.object(session_distill, "note_turn",
                           side_effect=RuntimeError("boom")):
        provider.sync_turn("u", "a", session_id="s")  # must not raise
    provider.shutdown()


# ── 5. shutdown cancels ─────────────────────────────────

def test_shutdown_cancels_pending_timer():
    rec = _RunRecorder()
    with mock.patch.object(session_distill.subprocess, "run", side_effect=rec):
        session_distill.note_turn()
        session_distill.cancel_scheduler()   # cancel before the 100ms window
        time.sleep(0.3)
        assert rec.calls == 0, "cancelled timer must not fire"
