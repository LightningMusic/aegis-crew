"""
aegis-crew -- CLI entry point.

Usage:
  python main.py "<your request>" --project /path/to/project
  python main.py --file requirements.md --project /path/to/project
  python main.py "<instruction>" --project /path/to/project --context-file docs/SRS

Examples:
  python main.py "write a script that renames all .jpg files in a folder
  to include today's date" --project C:\\Coding\\scratch\\rename-tool

  python main.py --file C:\\Coding\\my-app-srs.md --project C:\\Coding\\my-app

  python main.py "Implement the providers section as described in the SRS." \\
      --project C:\\Aquila-Experiment \\
      --context-file "C:\\Aquila-Experiment\\docs\\SRS\\Project-Aquila-SRS.md"

--context-file lets you give a short, specific instruction on the command
line (the "request") while attaching a larger reference document or SRS
directory as supporting context. This is different from --file, which
treats the given file/directory AS the entire request. --context-file is
appended after the request, clearly labeled, so the planner sees both
"what to do right now" and "what to do it against."

PLAN APPROVAL: after planning, the phase list (each phase's name,
description, and the exact files it's allowed to write to) is printed for
review before anything executes. This exists because a long local-model
conversation can drift off its original scope -- reviewing the plan up
front, rather than trusting the whole run end to end, is the actual
safeguard against that. Pass --auto-approve to skip this (e.g. for
scripted/unattended runs) and execute the plan as generated.

This is intentionally a plain CLI for now -- no UI, no WebSocket, no
browser layer that can silently swallow a message like AegisCoder's did.
Everything that happens is either printed to your terminal or written to
logs/aegis-crew.log and data/progress.json inside the project directory.
"""
import argparse
import logging
import sys
from pathlib import Path

from agents.config import LOG_DIR, MODEL_NAME
from agents.phase_planner import generate_clarifying_questions
from agents.pipeline import Phase, audit_project, execute_phase_plan, generate_project_plan
from safety import ollama_manager


def _setup_logging():
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "aegis-crew.log", encoding="utf-8"),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="aegis-crew -- local multi-agent coding pipeline (PM -> Dev/Infra -> Security -> Test)"
    )
    request_group = parser.add_mutually_exclusive_group(required=True)
    request_group.add_argument(
        "request", nargs="?", default=None,
        help="The request text directly on the command line."
    )
    request_group.add_argument(
        "--file", type=str, default=None,
        help="Path to a text/markdown request file, or a directory of SRS files. "
             "The file/directory contents ARE the request."
    )
    parser.add_argument(
        "--context-file", type=str, default=None,
        help="Path to a text/markdown file, or a directory of SRS files, to attach "
             "as supporting reference context alongside the positional request. "
             "Unlike --file, this does not replace the request -- it is appended "
             "to it, clearly labeled, so the planner has both the specific "
             "instruction and the larger document to work against. "
             "Not compatible with --file (which already reads its own source)."
    )
    parser.add_argument(
        "--project", type=str, required=True,
        help="Path to the project directory. Created if it doesn't exist."
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Force re-running all phases even if recorded as completed in progress.json."
    )
    parser.add_argument(
        "--no-questions", action="store_true", default=False,
        help="Skip the PM's interactive essential-questions check."
    )
    parser.add_argument(
        "--auto-approve", action="store_true", default=False,
        help="Skip the plan-review prompt and execute the generated plan as-is. "
             "Use for scripted/unattended runs; interactive runs should review the plan."
    )
    args = parser.parse_args()

    if args.context_file and args.file:
        parser.error("--context-file cannot be combined with --file; --file already reads its own source.")
    if args.context_file and not args.request:
        parser.error("--context-file requires a positional request (the specific instruction).")

    return args


def _read_request_source(source: Path) -> str:
    """Read one request file or combine supported SRS files from a directory."""
    if source.is_file():
        return source.read_text(encoding="utf-8", errors="replace")
    if not source.is_dir():
        raise FileNotFoundError(source)

    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".rst"}
    )
    if not files:
        raise ValueError(f"No .md, .txt, or .rst files found in {source}")
    return "\n\n".join(
        f"# Source: {path.relative_to(source).as_posix()}\n\n"
        + path.read_text(encoding="utf-8", errors="replace")
        for path in files
    )


def _collect_clarifications(request_text: str, project_path: str) -> str:
    """Ask essential PM questions only when an interactive terminal is available."""
    questions = generate_clarifying_questions(request_text, audit_project(project_path))
    if not questions:
        return request_text
    if not sys.stdin.isatty():
        logging.getLogger("main").warning("PM questions skipped because stdin is not interactive.")
        return request_text

    print("\nProject Manager needs a few decisions before planning:")
    answers: list[str] = []
    for question in questions:
        answer = input(f"- {question}\n  Answer (press Enter to leave open): ").strip()
        if answer:
            answers.append(f"Q: {question}\nA: {answer}")
    return request_text if not answers else request_text + "\n\nUSER CLARIFICATIONS:\n" + "\n\n".join(answers)


def _print_plan(phases: list[Phase]) -> None:
    print("\n" + "=" * 70)
    print(f"PROPOSED PLAN -- {len(phases)} phase(s)")
    print("=" * 70)
    for i, phase in enumerate(phases, 1):
        targets = phase.get("target_files") or []
        print(f"\n[{i}] {phase['name']}")
        print(f"    {phase['description']}")
        if targets:
            print(f"    Files this phase may write: {', '.join(targets)}")
        else:
            print("    Files this phase may write: (unrestricted -- no fixed file list)")
    print()


def _review_plan(phases: list[Phase]) -> list[Phase] | None:
    """
    Show the plan and let the user approve all, select a subset by number,
    or cancel. Returns the approved phase list, or None to abort the run.
    Falls through to approving everything if stdin isn't interactive, since
    there's no one there to answer -- matches _collect_clarifications'
    non-interactive behavior.
    """
    _print_plan(phases)
    if not sys.stdin.isatty():
        logging.getLogger("main").warning("Plan review skipped because stdin is not interactive -- running as planned.")
        return phases

    while True:
        choice = input(
            "Proceed with this plan? [y] run all phases  [s] select which phases to run  [n] cancel: "
        ).strip().lower()
        if choice in ("y", "yes", ""):
            return phases
        if choice in ("n", "no"):
            return None
        if choice in ("s", "select"):
            raw = input(
                f"Enter phase numbers to run, comma-separated (1-{len(phases)}): "
            ).strip()
            try:
                indices = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
            except ValueError:
                print("Couldn't parse that -- use numbers separated by commas, e.g. 1,3")
                continue
            selected = [phases[i - 1] for i in indices if 1 <= i <= len(phases)]
            if not selected:
                print("No valid phase numbers selected.")
                continue
            print(f"\nRunning {len(selected)} of {len(phases)} phase(s): "
                  f"{', '.join(p['name'] for p in selected)}")
            return selected
        print("Please answer y, s, or n.")


def main():
    _setup_logging()
    log = logging.getLogger("main")

    args = _parse_args()

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            log.error("Request file not found: %s", file_path)
            sys.exit(1)
        try:
            request_text = _read_request_source(file_path)
        except ValueError as exc:
            log.error("%s", exc)
            sys.exit(1)
        log.info("Loaded request from %s (%d chars)", file_path, len(request_text))
    else:
        request_text = args.request

    if not request_text or not request_text.strip():
        log.error("Empty request. Provide request text or --file.")
        sys.exit(1)

    if args.context_file:
        context_path = Path(args.context_file)
        if not context_path.exists():
            log.error("Context file/directory not found: %s", context_path)
            sys.exit(1)
        try:
            context_text = _read_request_source(context_path)
        except ValueError as exc:
            log.error("%s", exc)
            sys.exit(1)
        log.info("Loaded context from %s (%d chars)", context_path, len(context_text))
        request_text = (
            f"{request_text}\n\n"
            "SUPPORTING REFERENCE DOCUMENT (context, not the instruction itself -- "
            "use it to understand requirements/scope, but the instruction above "
            "is what to actually do):\n\n"
            f"{context_text}"
        )

    project_path = str(Path(args.project).resolve())
    log.info("Project directory: %s", project_path)

    log.info("Checking Ollama...")
    if not ollama_manager.ensure_running():
        log.error(
            "Could not start or reach Ollama. Check that 'ollama' is on PATH, "
            "or start it manually with: ollama serve"
        )
        sys.exit(1)

    if not ollama_manager.confirm_model_available():
        log.error("Model '%s' is not available. Run: ollama pull %s", MODEL_NAME, MODEL_NAME)
        sys.exit(1)

    log.info("Ollama ready, model '%s' confirmed available.", MODEL_NAME)

    try:
        if not args.no_questions:
            request_text = _collect_clarifications(request_text, project_path)

        phases = generate_project_plan(request_text, project_path)

        if args.auto_approve:
            approved_phases = phases
        else:
            approved_phases = _review_plan(phases)
            if approved_phases is None:
                log.info("Plan cancelled by user. Nothing was executed.")
                return

        execute_phase_plan(approved_phases, project_path, force=args.force, request_text=request_text)
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Progress made so far is saved in data/progress.json.")
        sys.exit(130)
    finally:
        ollama_manager.stop()

    log.info("Done. See %s for the full run log and %s/data/progress.json for phase-by-phase results.",
              Path(LOG_DIR) / "aegis-crew.log", project_path)


if __name__ == "__main__":
    main()