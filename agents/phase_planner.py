"""
aegis-crew -- phase planner.

Takes the user's full request -- anything from a one-line script ask to a
complete SRS document -- and breaks it into phases via a single direct call
to Ollama (not routed through AutoGen's GroupChat machinery, since this is
a one-shot planning call, not a conversation).

WHY A DIRECT CALL INSTEAD OF AN AGENT: AutoGen's OpenAI-compatible client
doesn't give us a clean way to set Ollama-specific options like num_ctx per
call. Calling Ollama's native /api/chat directly (same pattern AegisCoder's
architect.py used) lets us control that explicitly and keeps this planning
step simple and fast.

CONTEXT WINDOW HONESTY: local-code:7b has an 8192-token context window.
A genuinely massive SRS (dozens of pages) can still exceed what the model
can attend to in a single planning call, same as it always could with any
7B model. This planner does not chunk the SRS itself -- if you hit this
limit in practice (the phase list looks incomplete or generic), split the
SRS into sections yourself and run the planner once per section, or raise
NUM_CTX in the Modelfile if you have RAM headroom to spare. This module
does not pretend that limitation doesn't exist.
"""
import json
import logging
import re
from typing import Any

import httpx

from agents.config import MODEL_NAME, OLLAMA_API_BASE

log = logging.getLogger(__name__)

# Connect must succeed fast; generation gets real time for a large SRS.
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 300.0
_MAX_REQUEST_CONTEXT_CHARS = 14_000

PHASE_PLANNER_SYSTEM = """You are a software project planner. The user will
give you a request -- anything from a single script description to a full
SRS document for an entire application.

CRITICAL RULE FOR SIMPLE/SINGLE-SCRIPT REQUESTS:
If the request asks for a single script, utility, CLI tool, or small program (e.g., "Write a Python script that...", "Build a file converter...", "Create a tool..."), DO NOT split it into multiple phases. Produce EXACTLY ONE SINGLE PHASE. Single scripts belong in a single file unless explicitly requested otherwise.

For large multi-component applications (SRS documents, complex web apps, microservices), break the work into logical phases (e.g. Scaffolding, Core Logic, API, Testing).

When a project audit is supplied, treat it as the source of truth about work
already present. Plan only the gap: do not recreate existing files or
functionality, and explicitly extend the established foundation.

Output ONLY a JSON array, no preamble, no markdown fences, in this exact shape:
[{"name": "Short phase name", "description": "Specific tasks and files involved in this phase, in enough detail that a developer could start from this description alone."}]
"""

CLARIFY_SYSTEM = """You are a software project manager preparing to plan a
software change. Ask at most three questions only when their answers
materially change the implementation. Return ONLY a JSON array of question
strings; return [] when the provided request and audit are sufficient."""


def is_single_script_request(request_text: str) -> bool:
    text_lower = request_text.lower().strip()
    script_triggers = [
        "write a python script",
        "write a script",
        "create a script",
        "build a script",
        "write a tool",
        "single script",
        "one script",
        "a python script",
        "a script that",
    ]
    if any(trigger in text_lower for trigger in script_triggers):
        return True
    if len(request_text.strip()) < 300 and not any(kw in text_lower for kw in ["srs", "architecture", "microservice", "multi-module", "full stack", "phases"]):
        return True
    return False


def build_planning_prompt(request_text: str, project_audit: str = "") -> str:
    """Combine the SRS/request with read-only project-resume context."""
    request_context = _compact_request_context(request_text)
    if not project_audit:
        return request_context
    return (
        "USER REQUEST / SRS:\n"
        f"{request_context}\n\n"
        "PROJECT AUDIT (generated immediately before planning):\n"
        f"{project_audit}\n\n"
        "This project already has the files and completed work shown above. "
        "Only plan phases for what is NOT yet done. Do not recreate existing "
        "functionality — extend it."
    )


def _compact_request_context(request_text: str) -> str:
    """Keep large SRS inputs usable by the local model's finite context window."""
    if len(request_text) <= _MAX_REQUEST_CONTEXT_CHARS:
        return request_text
    head_size = _MAX_REQUEST_CONTEXT_CHARS // 2
    tail_size = _MAX_REQUEST_CONTEXT_CHARS - head_size
    return (
        request_text[:head_size]
        + "\n\n[SRS middle omitted from the planning prompt to fit the local model context. "
        "The project audit and live files remain available to implementation agents.]\n\n"
        + request_text[-tail_size:]
    )


def generate_phases(request_text: str, project_audit: str = "") -> list[dict[str, str]]:
    """
    Ask the model to break the request into phases.
    Returns a list of {"name": str, "description": str} dicts.
    Always returns at least one phase -- falls back to treating the whole
    request as a single phase if parsing fails or request is a single script.
    """
    log.info("Phase planner: breaking down request (%d chars)", len(request_text))

    if is_single_script_request(request_text):
        log.info("Phase planner: detected single-script request -- keeping as 1 phase")
        return [{"name": "Single-Script Implementation", "description": request_text.strip()}]

    raw = _call_model(build_planning_prompt(request_text, project_audit))
    phases = _parse_phases(raw)

    if len(request_text.strip()) < 400 and len(phases) > 3:
        log.info("Phase planner: collapsing %d over-split phases into 1 single phase for small request", len(phases))
        return [{"name": "Single-Script Implementation", "description": request_text.strip()}]

    log.info("Phase planner: produced %d phase(s)", len(phases))
    return phases


def generate_clarifying_questions(request_text: str, project_audit: str = "") -> list[str]:
    """Ask only questions whose answers materially change the implementation."""
    prompt = (
        "Before planning this software work, list at most three essential "
        "clarifying questions. Return ONLY a JSON array of strings. If the "
        "request and project audit are specific enough, return [].\n\n"
        + build_planning_prompt(request_text, project_audit)
    )
    raw = _call_model(prompt, system_prompt=CLARIFY_SYSTEM)
    try:
        parsed: Any = json.loads(raw[raw.find("[") : raw.rfind("]") + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    return [str(item).strip() for item in parsed if isinstance(item, str) and item.strip()][:3] if isinstance(parsed, list) else []


def _call_model(request_text: str, system_prompt: str = PHASE_PLANNER_SYSTEM) -> str:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request_text},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=10.0, pool=5.0)

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{OLLAMA_API_BASE}/api/chat", json=payload)
            r.raise_for_status()
            res_json: Any = r.json()
            if isinstance(res_json, dict):
                res_dict: dict[str, Any] = res_json  # pyright: ignore[reportUnknownVariableType]
                msg: Any = res_dict.get("message")
                if isinstance(msg, dict):
                    msg_dict: dict[str, Any] = msg  # pyright: ignore[reportUnknownVariableType]
                    content: Any = msg_dict.get("content")
                    if isinstance(content, str):
                        return content
            return ""
    except httpx.ConnectError:
        raise RuntimeError(
            "Cannot connect to Ollama -- is it running? "
            f"Checked {OLLAMA_API_BASE}."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Phase planning timed out after {_READ_TIMEOUT:.0f}s. "
            "The request may be very large, or the system is under heavy load."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}")


def _parse_phases(text: str) -> list[dict[str, str]]:
    phases = _try_json(text)
    if phases:
        return phases
    log.warning("Phase planner output could not be parsed as JSON -- treating whole request as one phase")
    return [{"name": "Full request", "description": text.strip()[:2000]}]


def _try_json(text: str) -> list[dict[str, str]]:
    text = re.sub(r"```json?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        result: Any = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    if not isinstance(result, list):
        return []

    phases: list[dict[str, str]] = []
    for item in result:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(item, dict):
            item_dict: dict[str, Any] = item  # pyright: ignore[reportUnknownVariableType]
            name_val: Any = item_dict.get("name")
            desc_val: Any = item_dict.get("description")
            if name_val is not None and desc_val is not None:
                phases.append({"name": str(name_val).strip(), "description": str(desc_val).strip()})
    return phases
