"""
aegis-crew -- Large-deletion guard.

Ported from AegisCoder, then corrected for a real failure mode: the
original version measured safety purely from unified-diff "-" line
counts against the OLD file's total line count. That works for a small
targeted edit, but it produces a false positive on any genuine full-file
rewrite -- if the new content shares no lines verbatim with the old
content (different formatting, restructured logic, renamed variables),
difflib.unified_diff correctly marks nearly every old line as removed
even though a complete, substantial replacement file is being written
right alongside it. The old check was blind to that replacement; it only
ever looked at the subtraction side of the diff.

The rule is now about actual resulting file size, not diff-line-matching:
  - Removing a function or a block, or rewriting the file's style/structure
    while keeping it a real file? Fine -- that's expected work, however
    the diff happens to render.
  - The new content leaving the file mostly empty relative to what it had
    before? That's a potential wipeout -- reject the write.

Implementation notes:
  We still parse unified diff format (as agents/tools.py builds it) to
  identify which file(s) are involved, but the actual safety decision
  now compares the OLD file's line count against the NEW content's line
  count (passed in directly by the caller, since tools.py already has
  both the old and new content in hand before it ever builds a diff).
  If a caller doesn't supply new_line_counts for a file (e.g. an older
  call site), we fall back to the previous diff-based deletion ratio so
  this remains safe to use without every caller being updated at once.
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


def check_diff(
    diff_text: str,
    project_path: str,
    new_line_counts: dict[str, int] | None = None,
) -> list[GuardResult]:
    """
    Parse a unified diff string and check every changed file against
    the deletion threshold.

    new_line_counts, when provided, maps relative file path -> the line
    count of the NEW content being written for that file. This is the
    preferred, accurate path: safety is judged by how much the file would
    actually shrink, not by how many old lines happen to match new lines
    verbatim. Pass this whenever the caller already has the new content
    in hand (agents/tools.py always does).

    Returns a list of GuardResult -- one per file in the diff.
    Call .safe on each to determine whether the edit is safe to apply.
    """
    new_line_counts = new_line_counts or {}
    results: list[GuardResult] = []
    current_file: str | None = None
    deletions: int = 0

    for line in diff_text.splitlines():
        # Detect file header: "--- a/path/to/file.py"
        if line.startswith("--- "):
            # Flush previous file if any
            if current_file is not None:
                results.append(_evaluate(current_file, deletions, project_path, new_line_counts.get(current_file)))
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
        results.append(_evaluate(current_file, deletions, project_path, new_line_counts.get(current_file)))

    return results


def _evaluate(filename: str, deletions: int, project_path: str, new_line_count: int | None = None) -> GuardResult:
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

    if new_line_count is not None:
        # Accurate path: judge safety by actual resulting file size, not by
        # how many old lines happen to match new lines verbatim. A full
        # rewrite that keeps the file a comparable, substantial size is
        # safe even if difflib thinks 100% of old lines were "removed" --
        # what matters is whether the file would end up mostly empty.
        shrink_ratio = max(0.0, 1.0 - (new_line_count / lines_total))
        safe = shrink_ratio <= DELETION_GUARD_THRESHOLD
        ratio = shrink_ratio
        basis = f"resulting file would have {new_line_count} of {lines_total} original lines' worth of content"
    else:
        # Legacy fallback for any call site that hasn't been updated to
        # pass new_line_counts -- same behavior as before.
        ratio = deletions / lines_total
        safe = ratio <= DELETION_GUARD_THRESHOLD
        basis = f"diff would remove {deletions} of {lines_total} lines"

    if not safe:
        message = (
            f"DELETION GUARD: {filename} -- {basis} ({ratio:.0%} shrink). "
            f"Threshold is {DELETION_GUARD_THRESHOLD:.0%}. "
            "Write rejected."
        )
        log.warning(message)
    else:
        message = f"{filename}: {basis} ({ratio:.0%} shrink) -- within threshold"

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