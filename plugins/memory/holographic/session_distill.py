"""Session-end LLM distillation hook for the holographic provider.

Replaces the English-regex ``_auto_extract_facts`` (which never matched
Chinese conversations) with the production LLM distillation pipeline
(``~/.hermes/scripts/distill_llm.py``): same pipeline as the cron/manual
full runs — the distill ledger provides idempotency, the built-in flock
prevents overlap, and failures are logged-and-forgotten (the daily cron
back-fills errored messages).

Design notes:
- Called from MemoryManager's background worker (``commit_session_boundary_async``
  docstring explicitly budgets "an LLM-bound call that can take seconds"),
  so a synchronous subprocess call here never blocks the conversation.
- One subprocess for the whole session's messages; the ledger dedups
  anything the cron already distilled.
- ``auto_extract`` must be truthy (schema default "false") — same gate as
  before, now opted in via config.
"""

import logging
import os
import shutil
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

# Single pipeline: the distill script IS the logic (no duplicate here).
_DISTILL_SCRIPT = os.path.expanduser("~/.hermes/scripts/distill_llm.py")
_VENV_PYTHON = "/usr/local/lib/hermes-agent/venv/bin/python3"
_SYSTEM_PYTHON = shutil.which("python3") or "/usr/bin/python3"
_HOOK_TIMEOUT_S = 600  # worker thread budget; cron back-fills on timeout

# ── sync_turn debounce scheduler ─────────────────────────────────────────
# sync_turn fires on EVERY completed turn, but a distill run per turn would
# (a) distill mid-conversation intermediate states and (b) stampede the LLM
# during busy sessions. Policy:
#   - remember the last turn time (cheap, lock-guarded)
#   - arm a one-shot timer for _DEBOUNCE_S: fires only if no new turn
#     arrived in the window (conversation went quiet → session likely
#     complete enough to distill)
#   - hard cap: even under continuous activity, at most one distill per
#     _HARD_CAP_S (active long sessions still get periodic coverage —
#     they never fire on_session_end under session_reset.mode: none)
#   - the timer runs the subprocess on its OWN thread: the memory
#     manager's serialized worker must never be held hostage for minutes
#     (its queue backs up per-turn syncs and prefetches otherwise)
#   - single-flight: _running guard + the distill script's own flock both
#     prevent overlapping runs; triggers during a run just re-arm
_DEBOUNCE_S = 300.0   # quiet window after the last turn
_HARD_CAP_S = 1800.0  # max interval between distills under constant activity

_sched_lock = threading.Lock()
_last_turn_ts = 0.0
_last_distill_ts = 0.0
_timer: threading.Timer | None = None
_running = False


def reset_scheduler() -> None:
    """Test hook: forget all scheduler state and cancel any pending timer."""
    global _timer, _running
    with _sched_lock:
        if _timer is not None:
            _timer.cancel()
        _timer = None
        _running = False
        global _last_turn_ts, _last_distill_ts
        _last_turn_ts = 0.0
        _last_distill_ts = 0.0


def cancel_scheduler() -> None:
    """Cancel the pending debounce timer (used on provider shutdown)."""
    with _sched_lock:
        global _timer
        if _timer is not None:
            _timer.cancel()
            _timer = None


def note_turn() -> None:
    """Record turn activity; (re)arm the debounced distill timer. Never blocks."""
    global _timer, _last_turn_ts, _last_distill_ts
    now = time.time()
    with _sched_lock:
        first_turn = _last_turn_ts == 0.0
        _last_turn_ts = now
        if _timer is not None:
            _timer.cancel()
        if first_turn:
            # Process-start anchor: never treat "never distilled in this
            # process" as "hard cap exceeded" — that would fire turn 1
            # immediately. Anchor the cap window at the first turn instead.
            _last_distill_ts = now
        if now - _last_distill_ts >= _HARD_CAP_S:
            # Hard cap exceeded since the last distill: run now (still on
            # the timer thread, still single-flight). Continuous activity
            # must never postpone distillation indefinitely.
            delay = 0.0
        else:
            delay = _DEBOUNCE_S
        _timer = threading.Timer(delay, _fire)
        _timer.daemon = True
        _timer.start()


def _fire() -> None:
    """Timer callback on its own thread: single-flight subprocess run."""
    global _running, _last_distill_ts
    with _sched_lock:
        if _running:
            return
        _running = True
    try:
        run_session_distillation([{"role": "user", "content": "turn"}])
        _last_distill_ts = time.time()
    finally:
        with _sched_lock:
            _running = False


def _python_exec() -> str:
    return _VENV_PYTHON if os.path.exists(_VENV_PYTHON) else _SYSTEM_PYTHON


def run_session_distillation(messages) -> int:
    """Trigger an incremental distill run. Returns the subprocess exit code.

    Never raises: session-end hooks must not fail the memory manager's
    serialized worker. A non-zero rc means the cron back-fill will retry
    the affected messages (ledger status 'error').
    """
    if not messages:
        return 0
    if not os.path.exists(_DISTILL_SCRIPT):
        logger.debug("distill script missing, skipping session distillation")
        return 0
    try:
        proc = subprocess.run(
            [_python_exec(), _DISTILL_SCRIPT, "distill", "--workers", "2"],
            capture_output=True,
            text=True,
            timeout=_HOOK_TIMEOUT_S,
        )
        if proc.returncode != 0:
            logger.warning(
                "session distillation rc=%s: %s",
                proc.returncode, (proc.stderr or proc.stdout or "")[-500:],
            )
        return proc.returncode
    except subprocess.TimeoutExpired:
        logger.warning("session distillation timed out (%ss); cron will retry",
                       _HOOK_TIMEOUT_S)
        return 124
    except Exception as exc:  # never propagate
        logger.warning("session distillation failed: %s", exc)
        return 1
