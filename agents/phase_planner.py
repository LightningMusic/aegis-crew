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

CONTEXT WINDOW HONESTY: this call requests PHASE_PLANNER_NUM_CTX tokens of
context explicitly (agents/config.py -- 32768 by default as of 2026-08-27,
matching a deliberate Modelfile fix; keep this in sync if the Modelfile's
own NUM_CTX changes, since a mismatch here silently overrides it back down
for this call specifically). A genuinely massive SRS (dozens or hundreds of
pages, or a project audit across hundreds of files) can still exceed even
that in a single planning call, same as it always could with any 7B model.
This planner does not chunk the SRS itself -- if you hit this limit in
practice (the phase list looks incomplete or generic), split the SRS into
sections yourself and run the planner once per section, or raise both
PHASE_PLANNER_NUM_CTX here and the Modelfile's own NUM_CTX together if you
have RAM headroom to spare. What this module DOES
do automatically is keep the two pieces that get folded into every
planning prompt -- the request/SRS text and the project audit -- each
under their own character budget (see agents/config.py), so a large audit
on a real project can no longer silently push a merely-large SRS past the
context window on its own; a request text was always compacted here, the
audit was not, and that asymmetry (an uncapped audit stacked on top of an
already near-cap request) is what produced a real timeout on a 9-file
provider rewrite job. This module still does not pretend the underlying
limitation doesn't exist -- it just no longer makes it worse than it has
to be.
"""
import json
import logging
import re
from typing import Any

import httpx

from agents.config import (
    MODEL_NAME,
    OLLAMA_API_BASE,
    PHASE_PLANNER_CONNECT_TIMEOUT_SECONDS,
    PHASE_PLANNER_MAX_AUDIT_CONTEXT_CHARS,
    PHASE_PLANNER_MAX_REQUEST_CONTEXT_CHARS,
    PHASE_PLANNER_NUM_CTX,
    PHASE_PLANNER_READ_TIMEOUT_SECONDS,
)

log = logging.getLogger(__name__)

PHASE_PLANNER_SYSTEM = """You are a software project planner. The user will
give you a request -- anything from a single script description to a full
SRS document for an entire application.

CRITICAL RULE FOR SIMPLE/SINGLE-SCRIPT REQUESTS:
If the request asks for a single script, utility, CLI tool, or small program (e.g., "Write a Python script that...", "Build a file converter...", "Create a tool..."), DO NOT split it into multiple phases. Produce EXACTLY ONE SINGLE PHASE. Single scripts belong in a single file unless explicitly requested otherwise.

For large multi-component applications (SRS documents, complex web apps, microservices), break the work into logical phases (e.g. Scaffolding, Core Logic, API, Testing).

WHEN A REQUEST NAMES SEVERAL INDEPENDENT FILES THAT SHARE ONE CONTRACT (e.g.
"rewrite every provider except base.py" across a list of provider files, or
any set of files that each independently implement the same interface):
give each file its OWN phase, one file per phase, rather than grouping
several into one. Each phase in this pipeline gets its own bounded,
fixed-length conversation (a hard cap on how many turns Developer, Infra,
Security, and Tester get together) -- a phase scoped to one file fits
comfortably inside that budget; a phase scoped to several files competes
for the same fixed number of turns across all of them and is more likely
to run out before every file is written, reviewed, and verified. This is
purely about phase GRANULARITY, not about skipping work -- every named
file still gets its own complete phase, just not bundled with the others.

When a project audit is supplied, treat it as the source of truth about work
already present. Plan only the gap: do not recreate existing files or
functionality, and explicitly extend the established foundation.

EVERY phase must include a "target_files" array: the exact relative file
paths (from the project root) that phase is allowed to create or modify.
Be specific and complete -- list every file the phase will touch, including
new files it needs to create. Do NOT include files that should only be
read for reference/style/contract (e.g. an existing file being used as a
template) -- those stay out of target_files and remain read-only for that
phase. If a phase is purely exploratory or read-only (e.g. "read these
files to understand the contract before the next phase"), use an empty
target_files array -- this is enforced as read-only, so that phase will
not be able to write anything at all. Only use an empty array when the
phase genuinely writes nothing; if it writes anything, list every file.

Output ONLY a JSON array, no preamble, no markdown fences, in this exact shape:
[{"name": "Short phase name", "description": "Specific tasks and files involved in this phase, in enough detail that a developer could start from this description alone.", "target_files": ["relative/path/one.py", "relative/path/two.py"]}]
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
    """Combine the SRS/request with read-only project-resume context.

    Both pieces are compacted independently before being combined -- see
    _compact_text. Previously only the request/SRS side was capped; a large
    but legitimate project audit (e.g. a real multi-file project's recursive
    listing plus progress.json) could still push the combined prompt to, or
    past, the model's context window on its own, with the request-side cap
    doing nothing to prevent it.
    """
    request_context = _compact_request_context(request_text)
    if not project_audit:
        return request_context
    audit_context = _compact_audit_context(project_audit)
    return (
        "USER REQUEST / SRS:\n"
        f"{request_context}\n\n"
        "PROJECT AUDIT (generated immediately before planning):\n"
        f"{audit_context}\n\n"
        "This project already has the files and completed work shown above. "
        "Only plan phases for what is NOT yet done. Do not recreate existing "
        "functionality — extend it."
    )


def _compact_text(text: str, max_chars: int, omission_note: str) -> str:
    """Shared head/tail-preserving compaction: keep the start (usually
    the most orienting content) and the end (usually the most recent/
    decision-relevant content, e.g. an audit's progress.json section)
    rather than truncating from one side only."""
    if len(text) <= max_chars:
        return text
    head_size = max_chars // 2
    tail_size = max_chars - head_size
    return text[:head_size] + omission_note + text[-tail_size:]


def _compact_request_context(request_text: str) -> str:
    """Keep large SRS inputs usable by the local model's finite context window."""
    return _compact_text(
        request_text,
        PHASE_PLANNER_MAX_REQUEST_CONTEXT_CHARS,
        "\n\n[SRS middle omitted from the planning prompt to fit the local model context. "
        "The project audit and live files remain available to implementation agents.]\n\n",
    )


def _compact_audit_context(audit_text: str) -> str:
    """Keep a large project audit (recursive file listing + progress.json)
    usable by the local model's finite context window. Head/tail-preserving
    like the request-side compaction, so the file listing's start and
    progress.json's content (always at the tail of audit_project's output)
    both survive even when the middle is cut."""
    return _compact_text(
        audit_text,
        PHASE_PLANNER_MAX_AUDIT_CONTEXT_CHARS,
        "\n\n[Audit middle omitted from the planning prompt to fit the local model context. "
        "The full audit is still used for other purposes; this is only a planning-prompt cap.]\n\n",
    )


def generate_phases(request_text: str, project_audit: str = "") -> list[dict[str, Any]]:
    """
    Ask the model to break the request into phases.
    Returns a list of {"name": str, "description": str, "target_files": list[str]} dicts.
    target_files is the enforced write-scope allowlist for that phase -- see
    pipeline._path_in_scope. An empty target_files list means unrestricted
    (used for single-script requests where a fixed file list doesn't apply).
    Always returns at least one phase -- falls back to treating the whole
    request as a single phase if parsing fails or request is a single script.
    """
    log.info("Phase planner: breaking down request (%d chars)", len(request_text))

    if is_single_script_request(request_text):
        log.info("Phase planner: detected single-script request -- keeping as 1 phase")
        return [{"name": "Single-Script Implementation", "description": request_text.strip(), "target_files": ["*"]}]

    raw = _call_model(build_planning_prompt(request_text, project_audit))
    phases = _parse_phases(raw)

    if len(request_text.strip()) < 400 and len(phases) > 3:
        log.info("Phase planner: collapsing %d over-split phases into 1 single phase for small request", len(phases))
        return [{"name": "Single-Script Implementation", "description": request_text.strip(), "target_files": ["*"]}]

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
        # num_ctx set explicitly rather than left to whatever the Modelfile
        # happens to bake in -- see PHASE_PLANNER_NUM_CTX in agents/config.py.
        "options": {"temperature": 0.2, "num_ctx": PHASE_PLANNER_NUM_CTX},
    }
    timeout = httpx.Timeout(
        connect=PHASE_PLANNER_CONNECT_TIMEOUT_SECONDS,
        read=PHASE_PLANNER_READ_TIMEOUT_SECONDS,
        write=10.0,
        pool=5.0,
    )

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
            f"Phase planning timed out after {PHASE_PLANNER_READ_TIMEOUT_SECONDS:.0f}s. "
            "The request may be very large, or the system is under heavy load. "
            "Raise PHASE_PLANNER_READ_TIMEOUT_SECONDS in .env if this keeps happening "
            "on legitimately large planning calls."
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}")


def _parse_phases(text: str) -> list[dict[str, Any]]:
    phases = _try_json(text)
    if phases:
        return phases
    log.warning("Phase planner output could not be parsed as JSON -- treating whole request as one phase")
    return [{"name": "Full request", "description": text.strip()[:2000], "target_files": ["*"]}]


def _try_json(text: str) -> list[dict[str, Any]]:
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

    phases: list[dict[str, Any]] = []
    for item in result:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(item, dict):
            item_dict: dict[str, Any] = item  # pyright: ignore[reportUnknownVariableType]
            name_val: Any = item_dict.get("name")
            desc_val: Any = item_dict.get("description")
            if name_val is not None and desc_val is not None:
                raw_targets: Any = item_dict.get("target_files", [])
                target_files: list[str] = (
                    [str(t).strip().replace("\\", "/") for t in raw_targets if str(t).strip()]
                    if isinstance(raw_targets, list) else []
                )
                phases.append({
                    "name": str(name_val).strip(),
                    "description": str(desc_val).strip(),
                    "target_files": target_files,
                })
    return phases