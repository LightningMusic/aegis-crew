"""
aegis-crew -- Ollama process manager.

Starts Ollama (if not already running) with resource constraints applied
AT LAUNCH, and provides a health check. This is a corrected port of
AegisCoder's ollama_manager.py, carrying forward two hard-won lessons:

1. BELOW_NORMAL priority is applied to the OLLAMA PROCESS ITSELF via
   creationflags, not to our own process. In AegisCoder, throttling the
   wrong process (the one serving the UI) caused the entire app to freeze
   whenever Ollama was busy, because the UI-serving process got starved
   of CPU by the OS scheduler. Here there is no UI-serving process to
   accidentally starve -- this is a CLI tool -- but the lesson still
   applies: throttle the process that does the heavy lifting, not the
   one you need to stay responsive.

2. The Windows Job Object memory cap is applied to Ollama specifically,
   and BEFORE we do anything else that could become a parent process in
   a job. Job Object membership is inherited by child processes spawned
   AFTER a process joins a job -- applying a cap to our own process first
   and then spawning Ollama as a child would have silently pulled Ollama
   into the same tight cap meant for a lightweight parent, causing a
   silent, unexplained death under memory pressure with no Python
   exception to catch. We avoid that entirely by never capping our own
   process at all -- only Ollama's.
"""
import logging
import os
import subprocess
import time

import httpx

from agents.config import (
    MODEL_NAME,
    OLLAMA_API_BASE,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_LOADED_MODELS,
    OLLAMA_MAX_MEMORY_MB,
    OLLAMA_NUM_PARALLEL,
    OLLAMA_NUM_THREAD,
)

from typing import Any

log = logging.getLogger(__name__)

_ollama_proc: subprocess.Popen[bytes] | None = None
_job_handle: Any = None


def is_running() -> bool:
    """Lightweight check -- does the Ollama API respond at all."""
    try:
        r = httpx.get(f"{OLLAMA_API_BASE}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def start() -> bool:
    """
    Start Ollama with resource constraints applied at launch. No-op
    (returns True) if Ollama is already running -- we never assume we
    own a pre-existing Ollama process, and never touch its limits.
    """
    global _ollama_proc

    if is_running():
        log.info("Ollama already running -- using existing instance")
        return True

    log.info("Starting Ollama with resource constraints...")

    env = {
        **os.environ,
        "OLLAMA_NUM_PARALLEL": OLLAMA_NUM_PARALLEL,
        "OLLAMA_MAX_LOADED_MODELS": OLLAMA_MAX_LOADED_MODELS,
        "OLLAMA_NUM_THREAD": OLLAMA_NUM_THREAD,
        "OLLAMA_KEEP_ALIVE": OLLAMA_KEEP_ALIVE,
    }

    creationflags = 0
    if os.name == "nt":
        # CREATE_NO_WINDOW: no visible console window.
        # BELOW_NORMAL_PRIORITY_CLASS: applied to Ollama -- the process
        # doing the actual CPU-heavy generation work -- so it yields to
        # the rest of the desktop rather than the other way around.
        creationflags = (
            subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )

    try:
        _ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        log.info("Ollama started (PID %d)", _ollama_proc.pid)
    except FileNotFoundError:
        log.error("ollama executable not found on PATH.")
        return False
    except Exception as exc:
        log.exception("Unexpected error starting Ollama: %s", exc)
        return False

    if os.name == "nt":
        _apply_memory_cap(_ollama_proc.pid, OLLAMA_MAX_MEMORY_MB)

    return True


def _apply_memory_cap(pid: int, max_memory_mb: int) -> None:
    """
    Apply a Windows Job Object memory cap directly to Ollama's PID.
    Graceful no-op if pywin32 isn't installed -- logs a warning and
    continues without the hard cap rather than failing the whole startup.
    """
    global _job_handle
    try:
        import win32job  # type: ignore[import-untyped] # pyright: ignore[reportMissingTypeStubs]
        import win32api  # type: ignore[import-untyped] # pyright: ignore[reportMissingTypeStubs]
        import win32con  # type: ignore[import-untyped] # pyright: ignore[reportMissingTypeStubs]
    except ImportError:
        log.warning(
            "pywin32 not installed -- Ollama memory cap not enforced. "
            "Install pywin32 for a hard memory ceiling."
        )
        return

    try:
        hJob = win32job.CreateJobObject(0, "AegisCrew-Ollama")  # pyright: ignore[reportArgumentType]
        info = win32job.QueryInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportArgumentType]
        max_bytes = max_memory_mb * 1024 * 1024

        if isinstance(info, dict) and "BasicLimitInformation" in info:
            basic: Any = info["BasicLimitInformation"]  # pyright: ignore[reportUnknownVariableType]
            if isinstance(basic, dict):
                basic["LimitFlags"] = basic.get("LimitFlags", 0) | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY  # pyright: ignore[reportUnknownMemberType]
        elif isinstance(info, dict):
            info["LimitFlags"] = info.get("LimitFlags", 0) | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY  # pyright: ignore[reportUnknownMemberType]
        if isinstance(info, dict):
            info["ProcessMemoryLimit"] = max_bytes

        win32job.SetInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation, info)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportArgumentType]

        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)  # pyright: ignore[reportUnknownMemberType]
        win32job.AssignProcessToJobObject(hJob, handle)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]

        _job_handle = hJob
        log.info("Ollama memory cap applied: %d MB", max_memory_mb)
    except Exception as exc:
        log.warning("Could not apply Ollama memory cap: %s -- continuing without it", exc)


def wait_for_ready(timeout: int = 30) -> bool:
    log.info("Waiting for Ollama to become ready (timeout %ds)...", timeout)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running():
            log.info("Ollama is ready")
            return True
        time.sleep(1.0)
    log.error("Ollama did not become ready within %ds", timeout)
    return False


def ensure_running() -> bool:
    """The single call the CLI entry point makes at startup."""
    if is_running():
        log.info("Ollama health check passed")
        return True
    if not start():
        return False
    return wait_for_ready()


def confirm_model_available() -> bool:
    """Check that MODEL_NAME is actually pulled before we try to use it --
    fails fast with a clear message instead of a confusing hang/timeout
    deep inside a phase conversation."""
    try:
        r = httpx.get(f"{OLLAMA_API_BASE}/api/tags", timeout=5.0)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if MODEL_NAME in names:
            return True
        log.error(
            "Model '%s' is not pulled. Available: %s. Run: ollama pull %s",
            MODEL_NAME, ", ".join(names) or "(none)", MODEL_NAME,
        )
        return False
    except Exception as exc:
        log.error("Could not check available models: %s", exc)
        return False


def stop():
    """Terminate Ollama only if we started it. Never touches a
    pre-existing Ollama instance we didn't spawn."""
    global _ollama_proc
    if _ollama_proc is not None:
        log.info("Stopping Ollama (PID %d)...", _ollama_proc.pid)
        try:
            _ollama_proc.terminate()
            _ollama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _ollama_proc.kill()
        except Exception as exc:
            log.warning("Error stopping Ollama: %s", exc)
        finally:
            _ollama_proc = None
