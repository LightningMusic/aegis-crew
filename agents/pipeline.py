"""
aegis-crew -- pipeline orchestration.

Top-level flow:
  1. phase_planner.generate_phases() breaks the user's request into phases
     (one direct model call, not a GroupChat).
  2. Each phase gets its OWN bounded GroupChat: Developer implements with
     Infra scaffolding files as needed, Security reviews, Tester verifies.
     Max rounds per phase is capped (MAX_GROUPCHAT_ROUNDS) so a
     disagreement can bounce back and forth but never loop forever
     unattended.
  3. Progress is checkpointed to disk after every phase (data/progress.json
     under the project directory), so a large multi-phase project can be
     inspected mid-run or picked back up rather than trusted as an
     all-or-nothing black box.

Only ONE phase conversation runs at a time -- this matches
OLLAMA_NUM_PARALLEL=1 exactly. There is never a moment where two phases
need the model simultaneously.
"""
import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Callable

import autogen

from agents.config import LLM_CONFIG, MAX_GROUPCHAT_ROUNDS, MAX_PHASE_CYCLES
from agents.personas import DEV_SYSTEM, INFRA_SYSTEM, SECURITY_SYSTEM, TEST_SYSTEM
from agents.phase_planner import generate_phases
from agents.tool_bridge import ROLE_TOOLS, TOOL_SCHEMA_DESCRIPTIONS, ToolFn, build_protocol_block, extract_tool_calls, make_interceptor
from agents.tools import list_files, make_dir, read_file, run_shell, run_tests, write_file
from safety import ollama_manager

log = logging.getLogger(__name__)

# A "phase" is always {"name": str, "description": str, "target_files": list[str]}.
Phase = dict[str, Any]
_AUDIT_IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "build", "dist"}
_AUDIT_MAX_CHARS = 24_000


def _progress_path(project_path: str) -> Path:
    p = Path(project_path) / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p / "progress.json"


def _load_progress(project_path: str) -> dict[str, Any]:
    path = _progress_path(project_path)
    if not path.exists():
        return {"phases": []}
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    except Exception:
        return {"phases": []}


def audit_project(project_path: str) -> str:
    """Produce a bounded, recursive read-only inventory for gap planning."""
    root = Path(project_path)
    files: list[str] = []
    if root.exists():
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            relative = item.relative_to(root)
            if any(part in _AUDIT_IGNORED_DIRS for part in relative.parts):
                continue
            files.append(relative.as_posix())

    progress_path = root / "data" / "progress.json"
    progress_text = "No existing data/progress.json."
    if progress_path.exists():
        progress_text = progress_path.read_text(encoding="utf-8", errors="replace")

    listing = "\n".join(sorted(files)) or "(no project files found)"
    audit = f"Recursive file listing:\n{listing}\n\nExisting data/progress.json:\n{progress_text}"
    if len(audit) > _AUDIT_MAX_CHARS:
        audit = audit[:_AUDIT_MAX_CHARS] + "\n[Audit truncated to preserve planner context.]"
    return audit


def _save_progress(project_path: str, progress: dict[str, Any]) -> None:
    _progress_path(project_path).write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _path_in_scope(path: str, allowed_paths: list[str]) -> bool:
    """
    True if `path` is inside this phase's declared write-scope.

    allowed_paths == ["*"] means explicitly unrestricted -- used only for
    the single-script fallback phases that phase_planner.py generates
    programmatically, where there's no fixed file list to give.

    An EMPTY allowed_paths list means READ-ONLY: no writes permitted at
    all. This used to mean "unrestricted" but that was backwards -- a
    planner-produced phase with no declared targets (e.g. "Read Reference
    Files") is exactly the kind of phase that should never be writing
    anything, and treating "nothing declared" as "anything goes" is what
    let a supposedly read-only phase scribble garbage files at the
    project root. If a phase legitimately needs to write files, the
    planner should be listing them.

    Otherwise a path is in scope if it exactly matches an allowed entry,
    or sits underneath an allowed entry that names a directory (with or
    without a trailing slash) -- this lets a phase say "src/bios/providers"
    and still be allowed to touch new files it creates inside that
    directory.
    """
    if allowed_paths == ["*"]:
        return True
    if not allowed_paths:
        return False
    norm = path.replace("\\", "/").lstrip("./")
    for allowed in allowed_paths:
        a = allowed.replace("\\", "/").rstrip("/").lstrip("./")
        if norm == a or norm.startswith(a + "/"):
            return True
    return False


def _resolve_scoped_path(path: str, allowed_paths: list[str]) -> tuple[str | None, bool]:
    """
    Resolve `path` against this phase's scope, tolerating a specific,
    confirmed local-model failure mode: local-code:7b reliably gets a
    multi-segment relative path right when calling read_file, then drops
    the directory prefix down to a bare filename when calling write_file
    for that SAME file moments later (observed: 'src/bios/providers/hp.py'
    on read_file, immediately followed by 'hp.py' on write_file, repeated
    154 times across three separate runs on six different provider files,
    even with the correct path restated in the kickoff message and in
    every rejection). This is not context loss or drift -- the model
    never gets the correct path wrong when reading it; it's a tool-call
    formatting failure specific to write_file once a second large
    argument (content) is also present. No amount of re-stating the
    instruction fixed it, so the fix lives here instead: if a rejected
    path's basename matches the basename of EXACTLY ONE file in this
    phase's declared scope, treat that as the intended file and use the
    full scoped path instead of failing the same way forever.

    Returns (resolved_path, was_corrected):
      - (path, False)          if path was already in scope, unchanged
      - (corrected_path, True) if path was out of scope but its basename
                                uniquely matched one scoped file
      - (None, False)          if path is genuinely out of scope (either
                                no basename match, or an ambiguous one --
                                never guess between two real candidates)
    """
    if _path_in_scope(path, allowed_paths):
        return path, False
    if not allowed_paths or allowed_paths == ["*"]:
        return None, False

    target_name = Path(path.replace("\\", "/")).name
    matches = [a for a in allowed_paths if Path(a.replace("\\", "/")).name == target_name]
    if len(matches) == 1:
        return matches[0], True
    return None, False


def _make_tool_wrappers(project_path: str, allowed_paths: list[str] | None = None) -> dict[str, ToolFn]:
    """
    Build properly-typed wrapper functions around agents/tools.py, each
    with `project_path` fixed via closure. AutoGen builds the JSON schema
    it hands to the model FROM these function signatures and their
    Annotated type hints. These wrappers are what actually get registered,
    on both the calling agents (for the schema) and the executor (for the
    real call) -- this is the ONLY path that goes through tools.py, which
    means it's the only path that goes through the deletion guard and the
    project-directory containment check. code_execution_config (raw code
    execution) is deliberately never used -- it would bypass every safety
    check in tools.py entirely.

    allowed_paths, when non-empty, is a hard write-scope allowlist for this
    phase (see _path_in_scope). This exists because a long-running local
    model can drift off its original instructions as a conversation grows
    past its context window -- relying on the system prompt alone to keep
    it inside the intended files is not enough. write_file and make_dir
    check against this list on every call; read_file and list_files stay
    unrestricted since reading is never destructive.
    """
    scope = allowed_paths or []

    def read_file_tool(
        path: Annotated[str, "Path to the file, relative to the project root."]
    ) -> str:
        return read_file(path=path, project_path=project_path)

    def write_file_tool(
        path: Annotated[str, "Path to the file, relative to the project root."],
        content: Annotated[str, "The full new content to write to the file."],
    ) -> str:
        resolved, corrected = _resolve_scoped_path(path, scope)
        if resolved is None:
            log.warning("write_file blocked: '%s' is outside this phase's scope %s", path, scope)
            if scope:
                return (
                    f"REJECTED: '{path}' is outside this phase's approved file scope. "
                    f"This phase may only write to: {', '.join(scope)}. "
                    "If this file genuinely needs to change, that belongs in a different phase -- "
                    "do not retry this path."
                )
            return (
                f"REJECTED: '{path}' -- this phase has no declared write scope, which means it is "
                "READ-ONLY. No file may be written in this phase. Use read_file / list_files instead."
            )
        if corrected:
            log.warning("write_file auto-corrected: '%s' -> '%s' (unique basename match in phase scope)", path, resolved)
        return write_file(path=resolved, content=content, project_path=project_path)

    def make_dir_tool(
        path: Annotated[str, "Directory path to create, relative to the project root."]
    ) -> str:
        resolved, corrected = _resolve_scoped_path(path, scope)
        if resolved is None:
            log.warning("make_dir blocked: '%s' is outside this phase's scope %s", path, scope)
            if scope:
                return (
                    f"REJECTED: '{path}' is outside this phase's approved file scope. "
                    f"This phase may only touch: {', '.join(scope)}."
                )
            return f"REJECTED: '{path}' -- this phase has no declared write scope and is READ-ONLY."
        if corrected:
            log.warning("make_dir auto-corrected: '%s' -> '%s' (unique basename match in phase scope)", path, resolved)
        return make_dir(path=resolved, project_path=project_path)

    def list_files_tool(
        path: Annotated[str, "Directory path to list, relative to the project root. Use '.' for the project root."]
    ) -> str:
        return list_files(path=path, project_path=project_path)

    def run_shell_tool(
        command: Annotated[str, "Shell command to run inside the project directory."]
    ) -> str:
        return run_shell(command=command, project_path=project_path)

    def run_tests_tool() -> str:
        return run_tests(project_path=project_path)

    return {
        "read_file": read_file_tool,
        "write_file": write_file_tool,
        "make_dir": make_dir_tool,
        "list_files": list_files_tool,
        "run_shell": run_shell_tool,
        "run_tests": run_tests_tool,
    }


def _register(
    caller: autogen.ConversableAgent,
    executor: autogen.ConversableAgent,
    tools: dict[str, ToolFn],
    tool_names: list[str],
) -> None:
    """Register each named tool's schema on `caller` (so the model knows
    it exists) and its real implementation on `executor` (so calling it
    actually runs the function). Descriptions come from
    tool_bridge.TOOL_SCHEMA_DESCRIPTIONS -- one definition, not duplicated
    here."""
    for name in tool_names:
        fn = tools[name]
        description = TOOL_SCHEMA_DESCRIPTIONS[name]
        caller.register_for_llm(name=name, description=description)(fn)
        executor.register_for_execution(name=name)(fn)


def _build_agents(
    project_path: str,
    execution_log: list[dict[str, Any]] | None = None,
    allowed_paths: list[str] | None = None,
) -> tuple[
    autogen.AssistantAgent,
    autogen.AssistantAgent,
    autogen.AssistantAgent,
    autogen.AssistantAgent,
    autogen.UserProxyAgent,
]:
    dev = autogen.AssistantAgent(
        "Developer", system_message=DEV_SYSTEM + build_protocol_block("Developer"), llm_config=LLM_CONFIG
    )
    infra = autogen.AssistantAgent(
        "Infra", system_message=INFRA_SYSTEM + build_protocol_block("Infra"), llm_config=LLM_CONFIG
    )
    security = autogen.AssistantAgent(
        "Security", system_message=SECURITY_SYSTEM + build_protocol_block("Security"), llm_config=LLM_CONFIG
    )
    tester = autogen.AssistantAgent(
        "Tester", system_message=TEST_SYSTEM + build_protocol_block("Tester"), llm_config=LLM_CONFIG
    )

    executor = autogen.UserProxyAgent(
        "Executor",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=MAX_GROUPCHAT_ROUNDS,
    )

    tools = _make_tool_wrappers(project_path, allowed_paths=allowed_paths)

    _register(dev, executor, tools, ROLE_TOOLS["Developer"])
    _register(infra, executor, tools, ROLE_TOOLS["Infra"])
    _register(security, executor, tools, ROLE_TOOLS["Security"])
    _register(tester, executor, tools, ROLE_TOOLS["Tester"])

    func_map: dict[str, Callable[..., Any]] = getattr(executor, "_function_map", {})  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
    executor.register_reply(
        [autogen.Agent, None],
        make_interceptor(func_map, execution_log=execution_log, role_tools=ROLE_TOOLS),
        position=0,
    )

    return dev, infra, security, tester, executor


_ROLLING_CONTEXT_FILES_MAX_CHARS = 2_500


def _build_rolling_context(project_path: str, progress: dict[str, Any], allowed_paths: list[str] | None = None) -> str:
    """
    Build a compact summary of live filesystem state and prior phase outcomes
    to pass as rolling context into the phase kickoff.

    IMPORTANT: on a real-sized project this file listing must stay small.
    local-code:7b's context window is tiny (8192 tokens). A prior run
    against a ~250-file project dumped every single file path into this
    section, on every phase and every retry cycle -- that listing alone
    was large enough to crowd out the actual phase instructions and file
    scope by the time the model got to generating a tool call, and the
    model started writing to real-but-wrong files it half-recognized from
    the flood (e.g. src/common/default.py instead of the intended
    src/bios/providers/default.py). When a phase has a declared scope
    (allowed_paths), we only ever need to show files inside the same
    directories as that scope, not the whole project -- so we do that
    instead, and hard-cap the result regardless as a defensive backstop.
    """
    p = Path(project_path)
    existing_files: list[str] = []
    if p.exists():
        for item in p.rglob("*"):
            if item.is_file():
                rel = item.relative_to(p).as_posix()
                if not any(rel.startswith(ignored) or f"/{ignored}" in rel for ignored in ("data", ".venv", "__pycache__", ".git", ".pytest_cache")):
                    existing_files.append(rel)

    if allowed_paths and allowed_paths != ["*"]:
        # Only show files that live in the same directories as this phase's
        # declared scope -- e.g. for src/bios/providers/*.py targets, show
        # what's currently in src/bios/providers/, not the whole project.
        relevant_dirs = {str(Path(t).parent.as_posix()) for t in allowed_paths}
        scoped_files = [
            f for f in existing_files
            if str(Path(f).parent.as_posix()) in relevant_dirs
        ]
        files_summary = ", ".join(sorted(scoped_files)) if scoped_files else "(none of the target files exist yet)"
    else:
        files_summary = ", ".join(sorted(existing_files)) if existing_files else "None (clean project directory)"

    if len(files_summary) > _ROLLING_CONTEXT_FILES_MAX_CHARS:
        files_summary = files_summary[:_ROLLING_CONTEXT_FILES_MAX_CHARS] + ", ... [truncated -- use list_files to see more]"

    completed_recaps: list[str] = []
    for ph in progress.get("phases", []):
        if ph.get("status") == "done":
            name = ph.get("name", "Unknown")
            sum_text = str(ph.get("summary") or "").replace("\n", " ")[:150]
            completed_recaps.append(f"Phase '{name}': {sum_text}")

    prior_summary = "\n".join(f"  - {r}" for r in completed_recaps) if completed_recaps else "None (this is the first phase)"

    return (
        f"PROJECT LIVE STATE & ROLLING CONTEXT:\n"
        f"- Relevant files currently on disk: {files_summary}\n"
        f"- Prior completed phases:\n{prior_summary}\n"
    )


def _is_passing_test_result(result: object) -> bool:
    """Return true only for the positive, structured result from run_tests."""
    return str(result).startswith("TESTS PASSED (SYNTAX OK")


def _verified_completion(execution_log: list[dict[str, Any]]) -> tuple[bool, str]:
    """Check the non-negotiable completion gate from executed tool evidence.

    A model's prose and a successful unrelated shell command never count as a
    pass. The last write must be followed by a successful `run_tests` call.
    """
    last_write = max(
        (index for index, entry in enumerate(execution_log)
         if entry.get("name") in {"write_file", "make_dir"}),
        default=-1,
    )
    passing_tests = [
        (index, str(entry.get("result", "")))
        for index, entry in enumerate(execution_log)
        if entry.get("name") == "run_tests"
        and entry.get("actor") == "Tester"
        and _is_passing_test_result(entry.get("result"))
    ]
    if not passing_tests:
        return False, "No successful run_tests result was recorded."
    test_index, test_result = passing_tests[-1]
    if test_index < last_write:
        return False, "Files changed after the last successful test run."
    return True, test_result


def _has_tested_completion_report(messages: list[dict[str, Any]]) -> bool:
    """Require Tester to leave usable, test-backed handoff instructions."""
    required_headings = ("What it does", "How to use it", "Verified behavior", "Test evidence")
    return any(
        msg.get("name") == "Tester"
        and all(heading in str(msg.get("content") or "") for heading in required_headings)
        for msg in messages
    )


def _run_phase_groupchat(
    phase: Phase, project_path: str, progress: dict[str, Any], repair_context: str = ""
) -> tuple[str, bool, int, str]:
    """Run one bounded GroupChat conversation for a single phase.
    Returns (summary, verified, tool_calls_executed_count, test_evidence)."""
    execution_log: list[dict[str, Any]] = []
    allowed_paths: list[str] = [str(p) for p in phase.get("target_files", []) if str(p).strip()]
    dev, infra, security, tester, executor = _build_agents(
        project_path, execution_log=execution_log, allowed_paths=allowed_paths
    )

    tool_reply_to: autogen.Agent | None = None

    def custom_speaker_selection(
        last_speaker: autogen.Agent, groupchat: autogen.GroupChat
    ) -> autogen.Agent | str:
        nonlocal tool_reply_to
        messages = groupchat.messages
        if not messages:
            return dev

        last_msg = messages[-1]
        content = str(last_msg.get("content") or "")

        if last_speaker != executor:
            calls = extract_tool_calls(content)
            if calls:
                speaker_name = getattr(last_speaker, "name", "Agent")
                log.info("Speaker selection: detected %d tool call(s) from %s -> selecting Executor next", len(calls), speaker_name)
                tool_reply_to = last_speaker
                return executor

        if last_speaker == executor:
            return tool_reply_to or dev

        return "auto"

    groupchat = autogen.GroupChat(
        agents=[dev, infra, security, tester, executor],
        messages=[],
        max_round=MAX_GROUPCHAT_ROUNDS,
        speaker_selection_method=custom_speaker_selection,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=LLM_CONFIG)

    rolling_context = _build_rolling_context(project_path, progress, allowed_paths=allowed_paths)

    if allowed_paths == ["*"]:
        scope_line = "FILE SCOPE FOR THIS PHASE: unrestricted (single-script request, no fixed file list).\n"
    elif allowed_paths:
        scope_line = (
            f"FILE SCOPE FOR THIS PHASE (hard-enforced -- writes outside this exact list are rejected, "
            f"no exceptions): {', '.join(allowed_paths)}\n"
        )
    else:
        scope_line = (
            "FILE SCOPE FOR THIS PHASE: none declared -- this phase is READ-ONLY. Any write_file or "
            "make_dir call will be rejected. Use read_file and list_files only.\n"
        )

    kickoff = (
        f"Phase: {phase['name']}\n\n"
        f"Description: {phase['description']}\n\n"
        f"Project directory: {project_path}\n\n"
        f"{rolling_context}\n"
        f"{repair_context}\n"
        f"{scope_line}\n"
        "Developer: begin implementation. Check existing files listed above and build into them "
        "whenever possible. Do NOT create duplicate files. Use write_file immediately to implement changes. "
        "Only write to paths in FILE SCOPE above -- re-read it before every write_file call."
    )

    executor.initiate_chat(manager, message=kickoff)

    messages = groupchat.messages or []
    summary = ""
    for msg in reversed(messages):
        c = str(msg.get("content") or "").strip()
        if c:
            summary = c
            break

    verified, test_evidence = _verified_completion(execution_log)
    if verified and not _has_tested_completion_report(messages):
        verified = False
        test_evidence = (
            "Tests passed, but Tester did not provide the required test-backed completion report "
            "(What it does, How to use it, Verified behavior, Test evidence)."
        )
    if not verified:
        log.warning("Phase '%s' is not verified: %s", phase["name"], test_evidence)
    return summary, verified, len(execution_log), test_evidence


def _print_final_summary(project_path: str, request_text: str) -> None:
    p = Path(project_path).resolve()
    py_files: list[str] = []
    for item in p.rglob("*.py"):
        rel = item.relative_to(p).as_posix()
        if not any(rel.startswith(ignored) or f"/{ignored}" in rel for ignored in ("data", ".venv", "__pycache__", ".git", ".pytest_cache")):
            py_files.append(rel)

    log.info("\n" + "=" * 70)
    log.info("FINAL PROJECT SUMMARY & USER INSTRUCTIONS")
    log.info("=" * 70)
    log.info("Original Request: %s", request_text)
    log.info("Project Directory: %s", project_path)
    log.info("Generated Python File(s): %s", ", ".join(sorted(py_files)) if py_files else "None")

    progress = _load_progress(project_path)
    for phase in progress.get("phases", []):
        phase_name = phase.get("name", "Unknown")
        if phase.get("status") == "done":
            log.info("TESTED COMPLETION REPORT for phase '%s':\n%s", phase_name, phase.get("summary", ""))
            log.info("Test evidence: %s", phase.get("test_evidence", ""))
        else:
            log.warning(
                "No completion instructions for unverified phase '%s': %s",
                phase_name,
                phase.get("test_evidence", "No test evidence recorded."),
            )

    log.info("=" * 70 + "\n")


def generate_project_plan(request_text: str, project_path: str) -> list[Phase]:
    """
    Audit the project and break the request into phases. Returns the phase
    list without executing anything -- the caller (main.py) is expected to
    show this to the user for approval before calling execute_phase_plan.
    """
    Path(project_path).mkdir(parents=True, exist_ok=True)

    project_audit = audit_project(project_path)
    log.info("Audited existing project before planning (%d chars).", len(project_audit))
    log.info("Planning phases for request (%d chars)...", len(request_text))
    phases: list[Phase] = generate_phases(request_text, project_audit=project_audit)
    log.info("Plan produced %d phase(s):", len(phases))
    for i, phase in enumerate(phases, 1):
        targets = phase.get("target_files") or []
        target_note = ", ".join(targets) if targets else "(unrestricted)"
        log.info("  Phase %d: %s -- targets: %s", i, phase["name"], target_note)
    return phases


def execute_phase_plan(phases: list[Phase], project_path: str, force: bool = False, request_text: str = "") -> None:
    """
    Run an already-generated (and presumably user-approved) phase list,
    one bounded agent conversation per phase, checkpointing progress to
    disk after every phase.
    """
    progress: dict[str, Any] = _load_progress(project_path)
    completed_names: set[str] = (
        set() if force else {
            p["name"] for p in progress.get("phases", []) if p and p.get("status") == "done"
        }
    )

    for i, phase in enumerate(phases, 1):
        if phase["name"] in completed_names:
            log.info("Skipping already-completed phase %d/%d: %s", i, len(phases), phase["name"])
            continue

        log.info("=" * 70)
        log.info("Starting phase %d/%d: %s", i, len(phases), phase["name"])
        log.info("=" * 70)

        started_at = time.time()
        summary = ""
        status = "incomplete"
        tool_count = 0
        test_evidence = "No test run recorded."
        repair_context = ""
        for cycle in range(1, MAX_PHASE_CYCLES + 1):
            # ensure_running() was previously only called once, at the very
            # start of main.py before phase 1. On a long multi-phase run,
            # if Ollama dies partway through (crash, manual restart, memory
            # cap, anything), nothing ever re-checked or restarted it -- the
            # pipeline just kept hammering a dead connection, burning through
            # an entire phase's retry budget in seconds and cascading into
            # every phase after it, until something outside this process
            # brought Ollama back. Checking here means a mid-run crash costs
            # one restart (a few seconds) instead of every remaining phase.
            if not ollama_manager.ensure_running():
                msg = "Ollama is not reachable and could not be restarted before this cycle."
                log.error("Phase %d cycle %d: %s", i, cycle, msg)
                summary = f"EXCEPTION: {msg}"
                verified = False
                test_evidence = summary
                if cycle < MAX_PHASE_CYCLES:
                    continue
                break

            try:
                summary, verified, tool_count, test_evidence = _run_phase_groupchat(
                    phase, project_path, progress, repair_context
                )
            except Exception as exc:
                log.exception("Phase %d (%s) cycle %d failed", i, phase["name"], cycle)
                summary = f"EXCEPTION: {exc}"
                verified = False
                test_evidence = summary

            if verified:
                status = "done"
                log.info("Phase %d verified on cycle %d with %d tool calls.", i, cycle, tool_count)
                break

            log.warning("Phase %d cycle %d/%d did not verify: %s", i, cycle, MAX_PHASE_CYCLES, test_evidence)
            repair_context = (
                "REPAIR CYCLE REQUIRED:\n"
                f"The previous bounded conversation did NOT verify this phase: {test_evidence}\n"
                "Regroup: inspect the current files, correct the failure, then have Tester run `run_tests`. "
                "Do not declare completion until that tool reports a pass after all final edits."
            )

        phase_record = {
            "name": phase["name"],
            "description": phase["description"],
            "target_files": phase.get("target_files", []),
            "status": status,
            "summary": summary[:2000],
            "test_evidence": test_evidence[:4000],
            "verification_cycles": cycle,
            "duration_seconds": round(time.time() - started_at, 1),
        }

        # Update existing phase entry or append
        existing_idx = None
        for idx, p in enumerate(progress.get("phases", [])):
            if p.get("name") == phase["name"]:
                existing_idx = idx
                break

        if existing_idx is not None:
            progress["phases"][existing_idx] = phase_record
        else:
            progress.setdefault("phases", []).append(phase_record)

        _save_progress(project_path, progress)

        log.info("Phase %d/%d finished: %s", i, len(phases), status)

    _print_final_summary(project_path, request_text)
    log.info("All phases processed. Progress recorded at %s", _progress_path(project_path))


def run_project(request_text: str, project_path: str, force: bool = False) -> None:
    """
    Non-interactive convenience wrapper: plan and execute in one call with
    no approval gate. Kept for any caller that wants the old all-in-one
    behavior (e.g. tests, or a future --auto-approve flag). Interactive use
    from main.py should call generate_project_plan(), show it to the user,
    and only call execute_phase_plan() after approval.
    """
    phases = generate_project_plan(request_text, project_path)
    execute_phase_plan(phases, project_path, force=force, request_text=request_text)