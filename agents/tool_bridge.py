"""
aegis-crew -- tool call protocol and interceptor.

WHY THIS EXISTS: local-code:7b, called through Ollama's OpenAI-compatible
endpoint, does not reliably return real structured tool_calls -- across
multiple test runs it consistently wrote {"name": "write_file", "arguments":
{...}} as PLAIN ASSISTANT TEXT instead of an actual API-level tool call.
AutoGen's default reply handling only executes a real tool call; text that
merely looks like one is invisible to it, which is why the Executor kept
replying with nothing and no files were ever written.

Rather than keep fighting a 7B model to speak a protocol it won't reliably
speak, this module recognizes the format it DOES reliably produce and
executes it directly. Two things make this reliable:
  1. We tell the model the EXACT JSON shape to write, explicitly, in its
     own system prompt (build_protocol_block) -- not just "use write_file"
     but the literal format we're going to parse for.
  2. We parse that format with a real JSON decoder (extract_tool_calls),
     not a naive regex, so file content containing braces/quotes doesn't
     break parsing.

This module is also the single source of truth for which role may call
which tool (ROLE_TOOLS), so pipeline.py (what's actually permitted at
dispatch time) and each persona's instructions (what the model is told)
can never drift apart from each other.

ON TYPING: AutoGen ships no type stubs of its own, so a reply function's
`recipient`/`sender` parameters (real agent instances at runtime) can only
be typed as `Any` here without depending on autogen's untyped internals.
Everything else in this file -- our own data shapes -- is fully typed.
"""
import json
from typing import Any, Callable

ToolFn = Callable[..., str]
ReplyFn = Callable[..., tuple[bool, str | None]]

# Which tools each role is permitted to call. pipeline.py registers exactly
# these for each agent -- change it here and both the registration and the
# persona's instructions update together.
ROLE_TOOLS: dict[str, list[str]] = {
    "Developer": ["read_file", "write_file", "list_files", "run_shell"],
    "Infra": ["read_file", "write_file", "make_dir", "list_files", "run_shell"],
    "Security": ["read_file", "list_files"],
    "Tester": ["read_file", "write_file", "list_files", "run_tests"],
}

# Short descriptions used when registering each tool's schema with AutoGen
# (register_for_llm) -- kept in case the model ever does emit a real
# tool_calls response; harmless if it doesn't.
TOOL_SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "read_file": "Read a file's contents.",
    "write_file": "Write content to a file. Creates the file if it doesn't exist. Rejected automatically if it would delete a large portion of an existing file's content.",
    "make_dir": "Create a directory, including any missing parent directories.",
    "list_files": "List the files and subdirectories inside a directory.",
    "run_shell": "Run a shell command inside the project directory and return its output.",
    "run_tests": "Run the project's test suite (pytest) and return pass/fail with output.",
}

# The exact literal JSON shape shown to the model for each tool, used to
# build each persona's tool-usage instructions. This is what
# extract_tool_calls() below is written to parse -- the two must stay in
# sync, which is why they live in the same file.
TOOL_USAGE_EXAMPLES: dict[str, str] = {
    "read_file": '{"name": "read_file", "arguments": {"path": "relative/path.py"}}'
                 " -- read a file's contents.",
    "write_file": '{"name": "write_file", "arguments": {"path": "relative/path.py", "content": "full file contents"}}'
                  " -- write a file. Creates it if it doesn't exist. Automatically REJECTED if it would"
                  " delete a large portion of an existing file -- if rejected, do not retry the same write,"
                  " reconsider the change instead.",
    "make_dir": '{"name": "make_dir", "arguments": {"path": "relative/dir"}}'
                " -- create a directory, including missing parents.",
    "list_files": '{"name": "list_files", "arguments": {"path": "."}}'
                  " -- list files/directories at a path relative to the project root ('.' for root).",
    "run_shell": '{"name": "run_shell", "arguments": {"command": "shell command"}}'
                 " -- run a shell command inside the project directory. Has a timeout; long-running"
                 " commands will be killed.",
    "run_tests": '{"name": "run_tests", "arguments": {}}'
                 " -- run the project's pytest suite and report pass/fail with output.",
}


def build_protocol_block(role: str) -> str:
    """
    Build the tool-usage instructions appended to a persona's system
    message, listing ONLY the tools that role is actually permitted to
    call (matches ROLE_TOOLS exactly, so the model is never told about a
    tool it will then be denied at dispatch time).
    """
    tool_names = ROLE_TOOLS.get(role, [])
    if not tool_names:
        return ""

    lines = [TOOL_USAGE_EXAMPLES[name] for name in tool_names]
    tool_list = "\n".join(f"  {line}" for line in lines)

    return f"""

TOOL USAGE:
To take an action, write a JSON object on its own line in EXACTLY this
shape (this is not a suggestion -- the executor parses this exact format):
  {{"name": "<tool_name>", "arguments": {{...}}}}

You may include multiple JSON objects in one message, one per line, to
take several actions at once. Available tools for your role:
{tool_list}

The Executor will run each one automatically and report the result back
to the conversation. Do not describe what you would do instead of doing
it -- emit the JSON and let it run.
"""


import json
import logging
import re
from typing import Any, Callable

log = logging.getLogger(__name__)

ToolFn = Callable[..., str]
ReplyFn = Callable[..., tuple[bool, str | None]]

ROLE_TOOLS: dict[str, list[str]] = {
    "Developer": ["read_file", "write_file", "list_files", "run_shell"],
    "Infra": ["read_file", "write_file", "make_dir", "list_files", "run_shell"],
    "Security": ["read_file", "list_files"],
    "Tester": ["read_file", "write_file", "list_files", "run_tests"],
}

TOOL_SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "read_file": "Read a file's contents.",
    "write_file": "Write content to a file. Creates the file if it doesn't exist. Rejected automatically if it would delete a large portion of an existing file's content.",
    "make_dir": "Create a directory, including any missing parent directories.",
    "list_files": "List the files and subdirectories inside a directory.",
    "run_shell": "Run a shell command inside the project directory and return its output.",
    "run_tests": "Run the project's test suite (pytest) and return pass/fail with output.",
}

TOOL_USAGE_EXAMPLES: dict[str, str] = {
    "read_file": '{"name": "read_file", "arguments": {"path": "relative/path.py"}}'
                 " -- read a file's contents.",
    "write_file": '{"name": "write_file", "arguments": {"path": "relative/path.py", "content": "full file contents"}}'
                  " -- write a file. Creates it if it doesn't exist. Automatically REJECTED if it would"
                  " delete a large portion of an existing file -- if rejected, do not retry the same write,"
                  " reconsider the change instead.",
    "make_dir": '{"name": "make_dir", "arguments": {"path": "relative/dir"}}'
                " -- create a directory, including missing parents.",
    "list_files": '{"name": "list_files", "arguments": {"path": "."}}'
                  " -- list files/directories at a path relative to the project root ('.' for root).",
    "run_shell": '{"name": "run_shell", "arguments": {"command": "shell command"}}'
                 " -- run a shell command inside the project directory. Has a timeout; long-running"
                 " commands will be killed.",
    "run_tests": '{"name": "run_tests", "arguments": {}}'
                 " -- run the project's pytest suite and report pass/fail with output.",
}


def build_protocol_block(role: str) -> str:
    """
    Build the tool-usage instructions appended to a persona's system
    message, listing ONLY the tools that role is actually permitted to
    call.
    """
    tool_names = ROLE_TOOLS.get(role, [])
    if not tool_names:
        return ""

    lines = [TOOL_USAGE_EXAMPLES[name] for name in tool_names]
    tool_list = "\n".join(f"  {line}" for line in lines)

    return f"""

TOOL USAGE:
To take an action, write a JSON object on its own line in EXACTLY this
shape (this is not a suggestion -- the executor parses this exact format):
  {{"name": "<tool_name>", "arguments": {{...}}}}

You may include multiple JSON objects in one message, one per line, to
take several actions at once. Available tools for your role:
{tool_list}

The Executor will run each one automatically and report the result back
to the conversation. Do not describe what you would do instead of doing
it -- emit the JSON and let it run.
"""


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """
    Scan a message for JSON objects shaped like {"name": ..., "arguments": {...}},
    tolerating variations in key names (e.g. tool, function, action, parameters, args)
    and falling back to code block parsing if needed.
    """
    calls: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        brace_idx = text.find("{", idx)
        if brace_idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace_idx)
            if isinstance(obj, dict):
                obj_dict: dict[str, Any] = obj  # pyright: ignore[reportUnknownVariableType]
                name = (
                    obj_dict.get("name")
                    or obj_dict.get("tool")
                    or obj_dict.get("function")
                    or obj_dict.get("action")
                )
                raw_args = (
                    obj_dict.get("arguments")
                    if "arguments" in obj_dict
                    else obj_dict.get("parameters")
                    if "parameters" in obj_dict
                    else obj_dict.get("args")
                    if "args" in obj_dict
                    else obj_dict.get("action_input")
                    if "action_input" in obj_dict
                    else obj_dict.get("input")
                )

                if name and raw_args is None:
                    raw_args = {
                        k: v
                        for k, v in obj_dict.items()
                        if k not in ("name", "tool", "function", "action")
                    }

                if isinstance(name, str):
                    if isinstance(raw_args, str):
                        try:
                            parsed = json.loads(raw_args)
                            if isinstance(parsed, dict):
                                raw_args = parsed
                        except Exception:
                            pass

                    args_dict: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
                    calls.append({"name": name, "arguments": args_dict})
            idx = max(end, brace_idx + 1)
        except json.JSONDecodeError:
            idx = brace_idx + 1

    if not calls:
        code_calls = _extract_code_blocks(text)
        calls.extend(code_calls)

    return calls


def _extract_code_blocks(text: str) -> list[dict[str, Any]]:
    """
    Fallback: if a model writes code in a markdown block without JSON, extract the target
    filename from:
      1. Comments in the first 3 lines of the code block (# file: script.py)
      2. The prose text preceding the code block ("Here is `counter.py`:")
      3. Python syntax detection fallback ("main.py")
    """
    calls: list[dict[str, Any]] = []
    pattern = r"```(?:python|bash|sh|json)?\s*\n(.*?)```"
    matches = list(re.finditer(pattern, text, re.DOTALL))

    for match in matches:
        content = match.group(1)
        if not content.strip():
            continue

        path_val = None

        # 1. Search inside first 3 lines of code block
        first_lines = content.strip().splitlines()[:3]
        for line in first_lines:
            m = re.search(r"#\s*(?:file(?:name)?|path):\s*([^\s]+)", line, re.IGNORECASE)
            if m:
                path_val = m.group(1).strip()
                break

        # 2. Search preceding prose text up to 300 chars before backticks
        if not path_val:
            start_pos = max(0, match.start() - 300)
            preceding = text[start_pos:match.start()]
            m = re.search(r"(?:file(?:name)?|path)?[:\s]*`?([\w\.-]+\.(?:py|json|toml|sh|txt|md|csv))`?", preceding, re.IGNORECASE)
            if m:
                path_val = m.group(1).strip()

        # 3. Default fallback if Python syntax is detected
        if not path_val and ("def " in content or "import " in content or "print(" in content or "with open" in content):
            path_val = "main.py"

        if path_val:
            calls.append({"name": "write_file", "arguments": {"path": path_val, "content": content}})

    return calls


def make_interceptor(
    function_map: dict[str, ToolFn],
    execution_log: list[dict[str, Any]] | None = None,
    role_tools: dict[str, list[str]] | None = None,
) -> ReplyFn:
    """
    Build an AutoGen-compatible reply function that executes any JSON tool calls
    found in the incoming message. Updates `execution_log` if provided.
    """

    def reply_func(
        recipient: Any,
        messages: list[dict[str, Any]] | None = None,
        sender: Any = None,
        config: Any = None,
    ) -> tuple[bool, str | None]:
        if not messages:
            return False, None

        content: str = str(messages[-1].get("content") or "")
        calls = extract_tool_calls(content)
        if not calls:
            return False, None

        results: list[str] = []
        # GroupChat routes through its manager, so `sender` can be
        # ``chat_manager`` rather than the agent that authored the message.
        # The message name is the authoritative role for permission checks.
        sender_name = str(messages[-1].get("name") or getattr(sender, "name", "Unknown"))
        for call in calls:
            name = call.get("name")
            arguments = call.get("arguments", {})

            if not isinstance(name, str) or name not in function_map:
                results.append(f"ERROR: '{name}' is not an available tool.")
                log.warning("Model requested unavailable tool '%s'", name)
                continue
            if role_tools is not None and name not in role_tools.get(sender_name, []):
                results.append(f"ERROR: {sender_name} is not permitted to call '{name}'.")
                log.warning("Blocked unauthorized tool '%s' from %s", name, sender_name)
                continue
            if not isinstance(arguments, dict):
                results.append(f"ERROR: arguments for '{name}' were not a JSON object.")
                continue

            fn = function_map[name]
            log.info("Interception: executing tool '%s' with args %s", name, {k: (v if k != "content" else f"<{len(str(v))} chars>") for k, v in arguments.items()})
            try:
                result = fn(**arguments)
            except TypeError as exc:
                result = f"ERROR: wrong arguments for '{name}': {exc}"
                log.error("Tool '%s' signature mismatch: %s", name, exc)
            except Exception as exc:
                result = f"ERROR: '{name}' raised an exception: {exc}"
                log.exception("Tool '%s' raised during execution", name)

            if execution_log is not None:
                execution_log.append(
                    {"actor": sender_name, "name": name, "arguments": arguments, "result": result}
                )

            results.append(f"{name} -> {result}")

        return True, "\n".join(results)

    return reply_func
