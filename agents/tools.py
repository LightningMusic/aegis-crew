"""
aegis-crew -- tool functions callable by agents.

Every file-write goes through the deletion guard. Every shell/test call
has a hard timeout so a runaway command can never hang the process
indefinitely. These functions are registered on the Executor UserProxyAgent
in pipeline.py -- agents call them by name, arguments come from the model's
tool-call output.
"""
import difflib
import logging
import subprocess
from pathlib import Path

from agents.config import SHELL_TIMEOUT_SECONDS, TEST_TIMEOUT_SECONDS
from safety.deletion_guard import any_unsafe, check_diff

log = logging.getLogger(__name__)


def _resolve_within_project(path: str, project_path: str) -> Path:
    """
    Resolve `path` relative to project_path and refuse to escape it.
    Supports both relative paths and absolute paths within project_path.
    """
    root = Path(project_path).resolve()
    p = Path(path)
    if p.is_absolute():
        try:
            rel = p.relative_to(root)
            candidate = (root / rel).resolve()
        except ValueError:
            raise PermissionError(
                f"Refused: '{path}' resolves outside the project directory ({root}). "
                "All file operations must stay within the project."
            )
    else:
        candidate = (root / path).resolve()

    try:
        candidate.relative_to(root)
    except ValueError:
        raise PermissionError(
            f"Refused: '{path}' resolves outside the project directory ({root}). "
            "All file operations must stay within the project."
        )
    return candidate


def read_file(path: str, project_path: str) -> str:
    """Read a file's contents. Returns an error string instead of raising,
    since these functions are called by agents and need to hand back
    something the model can react to."""
    try:
        target = _resolve_within_project(path, project_path)
        if not target.exists():
            return f"ERROR: {path} does not exist."
        return target.read_text(encoding="utf-8", errors="replace")
    except PermissionError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:
        return f"ERROR reading {path}: {exc}"


def write_file(path: str, content: str, project_path: str) -> str:
    """
    Write content to a file, guarded against large accidental deletions.
    If the file already exists and this write would remove more than the
    configured threshold of its existing lines, the write is REJECTED and
    nothing is touched on disk.
    """
    try:
        target = _resolve_within_project(path, project_path)
    except PermissionError as exc:
        return f"ERROR: {exc}"

    if target.exists():
        try:
            old = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"ERROR reading existing {path} before write: {exc}"

        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        results = check_diff(diff, project_path, new_line_counts={path: len(content.splitlines())})
        if any_unsafe(results):
            message = next(r.message for r in results if not r.safe)
            log.warning("write_file rejected for %s: %s", path, message)
            return f"REJECTED: {message}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"ERROR writing {path}: {exc}"

    log.info("Wrote %d chars to %s", len(content), path)
    return f"OK: wrote {len(content)} chars to {path}"


def make_dir(path: str, project_path: str) -> str:
    """
    Create a directory (and any missing parents). Infra-agent tool --
    the Developer agent is not given this tool, so directory creation is
    enforced as an Infra responsibility rather than left to persona text.
    """
    try:
        target = _resolve_within_project(path, project_path)
        target.mkdir(parents=True, exist_ok=True)
        return f"OK: created directory {path}"
    except PermissionError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:
        return f"ERROR creating directory {path}: {exc}"


def list_files(path: str, project_path: str) -> str:
    """List files and directories under `path` (non-recursive). Useful for
    agents to check what already exists before creating or editing."""
    try:
        target = _resolve_within_project(path, project_path)
        if not target.exists():
            return f"ERROR: {path} does not exist."
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(entries) if entries else "(empty directory)"
    except PermissionError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:
        return f"ERROR listing {path}: {exc}"


def run_shell(command: str, project_path: str) -> str:
    """
    Run a shell command inside the project directory with a hard timeout.
    Output is truncated to keep it from blowing the model's context window
    on a chatty command.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output[:4000]
        return f"exit_code={result.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {SHELL_TIMEOUT_SECONDS}s and was killed."
    except Exception as exc:
        return f"ERROR running command: {exc}"


def run_tests(project_path: str) -> str:
    """
    Run py_compile syntax check on all Python files and execute pytest.
    Used exclusively by the Tester agent.
    """
    import py_compile
    import sys
    root = Path(project_path).resolve()

    syntax_errors: list[str] = []
    for py_file in root.rglob("*.py"):
        rel_parts = py_file.relative_to(root).parts
        if any(part in rel_parts for part in (".venv", "__pycache__", ".git", "data", "build")):
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as exc:
            rel_path = py_file.relative_to(root).as_posix()
            # PyCompileError stores the original SyntaxError in exc_value;
            # it does not itself expose a lineno attribute on all supported
            # Python versions.  Never let failure reporting hide a failure.
            syntax_error = getattr(exc, "exc_value", None)
            line_no = getattr(syntax_error, "lineno", None) or "?"
            syntax_errors.append(f"SYNTAX ERROR in {rel_path} (line {line_no}): {exc.msg}")

    if syntax_errors:
        return "TESTS FAILED (SYNTAX ERRORS DETECTED):\n" + "\n".join(syntax_errors)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "--tb=short"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output[:4000]
        if result.returncode == 0:
            return f"TESTS PASSED (SYNTAX OK & PYTEST PASSED):\n{output}"
        elif result.returncode == 5:
            return "TESTS PASSED (SYNTAX CHECK OK - All Python files compiled cleanly with zero syntax errors)."
        else:
            return f"TESTS FAILED (PYTEST FAILED exit_code={result.returncode}):\n{output}"
    except subprocess.TimeoutExpired:
        return f"ERROR: tests timed out after {TEST_TIMEOUT_SECONDS}s and were killed."
    except Exception as exc:
        return f"SYNTAX CHECK OK (No syntax errors found). pytest execution message: {exc}"