"""
aegis-crew -- CLI entry point.

Usage:
  python main.py "<your request>" --project /path/to/project
  python main.py --file requirements.md --project /path/to/project

Examples:
  python main.py "write a script that renames all .jpg files in a folder
  to include today's date" --project C:\\Coding\\scratch\\rename-tool

  python main.py --file C:\\Coding\\my-app-srs.md --project C:\\Coding\\my-app

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
from agents.pipeline import audit_project, run_project
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
        help="Path to a text/markdown request file, or a directory of SRS files."
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
    return parser.parse_args()


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
        run_project(request_text, project_path, force=args.force)
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Progress made so far is saved in data/progress.json.")
        sys.exit(130)
    finally:
        ollama_manager.stop()

    log.info("Done. See %s for the full run log and %s/data/progress.json for phase-by-phase results.",
              Path(LOG_DIR) / "aegis-crew.log", project_path)


if __name__ == "__main__":
    main()
