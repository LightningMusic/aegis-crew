import pytest
from agents.config import LLM_CONFIG, MAX_GROUPCHAT_ROUNDS
from agents.phase_planner import _compact_request_context, _parse_phases, build_planning_prompt, generate_phases
from agents.tool_bridge import extract_tool_calls, make_interceptor
from agents.pipeline import _has_tested_completion_report, _is_passing_test_result, _verified_completion, audit_project
from agents.tools import run_tests
from main import _read_request_source


def test_extract_tool_calls():
    # Standard JSON
    sample = 'Here is a call: {"name": "write_file", "arguments": {"path": "foo.py", "content": "print(1)"}}'
    calls = extract_tool_calls(sample)
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"

    # Field name variations (tool / parameters)
    sample_var = '{"tool": "make_dir", "parameters": {"path": "src"}}'
    calls_var = extract_tool_calls(sample_var)
    assert len(calls_var) == 1
    assert calls_var[0]["name"] == "make_dir"
    assert calls_var[0]["arguments"] == {"path": "src"}

    # Markdown code block fallback
    code_block = "```python\n# filename: app.py\nprint('hello')\n```"
    calls_code = extract_tool_calls(code_block)
    assert len(calls_code) == 1
    assert calls_code[0]["name"] == "write_file"
    assert calls_code[0]["arguments"]["path"] == "app.py"


def test_make_interceptor():
    def mock_fn(path: str) -> str:
        return f"read {path}"

    function_map = {"read_file": mock_fn}
    exec_log = []
    interceptor = make_interceptor(function_map, execution_log=exec_log)

    # Message with tool call
    msgs = [{"content": '{"name": "read_file", "arguments": {"path": "test.txt"}}'}]
    handled, reply = interceptor(None, msgs)
    assert handled is True
    assert reply == "read_file -> read test.txt"
    assert len(exec_log) == 1

    # Message without tool call
    msgs_none = [{"content": "Just talking"}]
    handled_none, reply_none = interceptor(None, msgs_none)
    assert handled_none is False
    assert reply_none is None


def test_parse_phases():
    raw_json = '[{"name": "P1", "description": "Desc 1"}]'
    phases = _parse_phases(raw_json)
    assert len(phases) == 1
    assert phases[0]["name"] == "P1"


def test_completion_requires_a_tester_test_after_final_write():
    passing = "TESTS PASSED (SYNTAX OK & PYTEST PASSED):\n1 passed"
    verified, evidence = _verified_completion([
        {"actor": "Developer", "name": "write_file", "result": "OK"},
        {"actor": "Tester", "name": "run_tests", "result": passing},
    ])
    assert verified is True
    assert evidence == passing

    verified_after_write, reason = _verified_completion([
        {"actor": "Tester", "name": "run_tests", "result": passing},
        {"actor": "Developer", "name": "write_file", "result": "OK"},
    ])
    assert verified_after_write is False
    assert reason == "Files changed after the last successful test run."

    verified_wrong_actor, reason = _verified_completion([
        {"actor": "Developer", "name": "run_tests", "result": passing},
    ])
    assert verified_wrong_actor is False
    assert reason == "No successful run_tests result was recorded."
    assert _is_passing_test_result(passing) is True


def test_run_tests_catches_unclosed_parenthesis_before_pytest(tmp_path):
    (tmp_path / "broken.py").write_text("value = (\n", encoding="utf-8")

    result = run_tests(str(tmp_path))

    assert result.startswith("TESTS FAILED (SYNTAX ERRORS DETECTED):")
    assert "SYNTAX ERROR in broken.py (line 1)" in result


def test_completion_report_must_be_from_tester_and_include_all_sections():
    report = "What it does\nHow to use it\nVerified behavior\nTest evidence"
    assert _has_tested_completion_report([{"name": "Tester", "content": report}]) is True
    assert _has_tested_completion_report([{"name": "Developer", "content": report}]) is False


def test_project_audit_is_recursive_and_excludes_dependency_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "base.py").write_text("class Base: pass", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "hidden.py").write_text("ignored", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "progress.json").write_text('{"phases": [{"status": "done"}]}', encoding="utf-8")

    audit = audit_project(str(tmp_path))

    assert "src/base.py" in audit
    assert ".venv/hidden.py" not in audit
    assert '"status": "done"' in audit


def test_planning_prompt_explicitly_requests_gap_only_plan():
    prompt = build_planning_prompt("Build the missing providers.", "Recursive file listing:\nsrc/base.py")
    assert "Build the missing providers." in prompt
    assert "Only plan phases for what is NOT yet done" in prompt
    assert "src/base.py" in prompt


def test_request_source_accepts_an_srs_directory(tmp_path):
    (tmp_path / "a.md").write_text("First requirement", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("Second requirement", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("not SRS", encoding="utf-8")

    request = _read_request_source(tmp_path)

    assert "# Source: a.md" in request
    assert "# Source: nested/b.txt" in request
    assert "not SRS" not in request


def test_large_srs_is_compacted_without_discarding_its_start_or_end():
    source = "START" + ("x" * 20_000) + "END"
    compacted = _compact_request_context(source)

    assert len(compacted) < len(source)
    assert compacted.startswith("START")
    assert compacted.endswith("END")
    assert "SRS middle omitted" in compacted
