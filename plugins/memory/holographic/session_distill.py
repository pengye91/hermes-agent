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

logger = logging.getLogger(__name__)

# Single pipeline: the distill script IS the logic (no duplicate here).
_DISTILL_SCRIPT = os.path.expanduser("~/.hermes/scripts/distill_llm.py")
_VENV_PYTHON = "/usr/local/lib/hermes-agent/venv/bin/python3"
_SYSTEM_PYTHON = shutil.which("python3") or "/usr/bin/python3"
_HOOK_TIMEOUT_S = 600  # worker thread budget; cron back-fills on timeout


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
