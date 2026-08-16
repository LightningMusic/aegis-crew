"""
aegis-crew -- Large-deletion guard.

Ported directly from the AegisCoder project. The logic is framework-agnostic:
before any diff is applied, this module checks what fraction of the
existing file's lines would be removed. If the fraction exceeds
DELETION_GUARD_THRESHOLD the edit is flagged and returned to the caller
with a warning rather than applied.

The rule is simple:
  - Removing a function or a block? Fine -- probably expected.
  - Removing 40%+ of a file? That's a potential wipeout -- reject the write.

Implementation notes:
  We parse unified diff format (the output difflib.unified_diff produces,
  which is what agents/tools.py builds before calling check_diff).
  Lines starting with "-" (not "---") are deletions.
  We compare deletion count against the original file's total line count.
  If the file does not exist on disk yet (new file creation), no check is
  needed and the guard always returns safe.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

from agents.config import DELETION_GUARD_THRESHOLD

log = logging.getLogger(__name__)


@dataclass
class GuardResult:
    safe: bool
    filename: str
    lines_total: int
    lines_deleted: int
    deletion_ratio: float
    threshold: float
    message: str


def check_diff(diff_text: str, project_path: str) -> list[GuardResult]:
    """
    Parse a unified diff string and check every changed file against
    the deletion threshold.

    Returns a list of GuardResult -- one per file in the diff.
    Call .safe on each to determine whether the edit is safe to apply.
    """
    results: list[GuardResult] = []
    current_file: str | None = None
    deletions: int = 0

    for line in diff_text.splitlines():
        # Detect file header: "--- a/path/to/file.py"
        if line.startswith("--- "):
            # Flush previous file if any
            if current_file is not None:
                results.append(_evaluate(current_file, deletions, project_path))
            # Parse the filename from the header
            parts = line[4:].strip()
            if parts.startswith("a/"):
                parts = parts[2:]
            current_file = parts
            deletions = 0

        elif line.startswith("---") and current_file is None:
            pass  # diff header before any file -- ignore

        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    # Flush last file
    if current_file is not None:
        results.append(_evaluate(current_file, deletions, project_path))

    return results


def _evaluate(filename: str, deletions: int, project_path: str) -> GuardResult:
    """Build a GuardResult for one file."""
    filepath = Path(project_path) / filename

    # New file -- nothing to delete from, always safe
    if not filepath.exists():
        return GuardResult(
            safe=True,
            filename=filename,
            lines_total=0,
            lines_deleted=0,
            deletion_ratio=0.0,
            threshold=DELETION_GUARD_THRESHOLD,
            message="New file -- no deletion check needed",
        )

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines_total = sum(1 for _ in fh)
    except OSError as exc:
        log.warning("Could not read %s to check deletion ratio: %s", filepath, exc)
        return GuardResult(
            safe=True,
            filename=filename,
            lines_total=0,
            lines_deleted=deletions,
            deletion_ratio=0.0,
            threshold=DELETION_GUARD_THRESHOLD,
            message=f"Could not read file: {exc}",
        )

    if lines_total == 0:
        return GuardResult(
            safe=True,
            filename=filename,
            lines_total=0,
            lines_deleted=deletions,
            deletion_ratio=0.0,
            threshold=DELETION_GUARD_THRESHOLD,
            message="File is empty -- no deletion check needed",
        )

    ratio = deletions / lines_total
    safe = ratio <= DELETION_GUARD_THRESHOLD

    if not safe:
        message = (
            f"DELETION GUARD: {filename} -- diff would remove {deletions} of "
            f"{lines_total} lines ({ratio:.0%} of file). "
            f"Threshold is {DELETION_GUARD_THRESHOLD:.0%}. "
            "Write rejected."
        )
        log.warning(message)
    else:
        message = (
            f"{filename}: {deletions}/{lines_total} lines deleted "
            f"({ratio:.0%}) -- within threshold"
        )

    return GuardResult(
        safe=safe,
        filename=filename,
        lines_total=lines_total,
        lines_deleted=deletions,
        deletion_ratio=ratio,
        threshold=DELETION_GUARD_THRESHOLD,
        message=message,
    )


def any_unsafe(results: list[GuardResult]) -> bool:
    """Convenience: True if any file in the diff triggered the guard."""
    return any(not r.safe for r in results)
