"""
aegis-crew -- central configuration.

Every tuneable value in this project lives here. Nothing is hardcoded
elsewhere. Defaults are conservative for a 16GB, no-GPU laptop.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Ollama connection
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_API_BASE = f"http://{OLLAMA_HOST}"
# AutoGen speaks the OpenAI HTTP API shape. Ollama exposes an
# OpenAI-compatible surface under /v1.
OLLAMA_OPENAI_BASE = f"{OLLAMA_API_BASE}/v1"

MODEL_NAME = os.getenv("MODEL_NAME", "local-code:7b")

# ---------------------------------------------------------------------------
# Ollama resource constraints (applied when we start Ollama ourselves --
# see safety/ollama_manager.py)
# ---------------------------------------------------------------------------
OLLAMA_NUM_PARALLEL = os.getenv("OLLAMA_NUM_PARALLEL", "1")
OLLAMA_MAX_LOADED_MODELS = os.getenv("OLLAMA_MAX_LOADED_MODELS", "1")
OLLAMA_NUM_THREAD = os.getenv("OLLAMA_NUM_THREAD", "8")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
OLLAMA_MAX_MEMORY_MB = int(os.getenv("OLLAMA_MAX_MEMORY_MB", "10240"))

# ---------------------------------------------------------------------------
# AutoGen LLM config -- every agent persona points at the SAME model.
# This is the deliberate RAM-optimization choice: five roles, one loaded
# model, zero extra memory cost versus a single-agent setup.
# ---------------------------------------------------------------------------
from typing import Any

# AutoGen's OpenAI-compatible client defaults to a ~600s request timeout if
# none is given. On a 16GB, no-GPU laptop running local-code:7b, a single
# completion over a large, growing conversation (an SRS-sized initial
# request plus every subsequent tool result) can legitimately take longer
# than that -- the model isn't hung, it's just slow. Without an explicit
# timeout here, AutoGen gives up and reports "OpenAI API call timed out"
# even though Ollama is still working, which wastes the whole phase.
# Raise this further via .env if you still see timeouts on large phases.
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "1800"))

LLM_CONFIG: dict[str, Any] = {
    "config_list": [
        {
            "model": MODEL_NAME,
            "base_url": OLLAMA_OPENAI_BASE,
            "api_key": "ollama",  # unused by Ollama but required by the OpenAI client shape
            "timeout": LLM_REQUEST_TIMEOUT_SECONDS,
        }
    ],
    "temperature": 0.2,
    "cache_seed": None,  # disable AutoGen's response cache -- always want fresh reasoning
    "timeout": LLM_REQUEST_TIMEOUT_SECONDS,
}

# ---------------------------------------------------------------------------
# Phase planner (direct Ollama /api/chat call, not routed through AutoGen --
# see agents/phase_planner.py for why). Every tuneable value in this project
# lives here per the rule at the top of this file; phase_planner.py used to
# hardcode its own timeout and context-budget constants locally, which is
# how a large SRS-backed planning call ended up on a 300s timeout instead
# of the same generous budget every other model call in this project gets
# (LLM_REQUEST_TIMEOUT_SECONDS above) -- that mismatch is what caused a
# real timed-out run on a 9-file provider rewrite job.
# ---------------------------------------------------------------------------
PHASE_PLANNER_CONNECT_TIMEOUT_SECONDS = float(os.getenv("PHASE_PLANNER_CONNECT_TIMEOUT_SECONDS", "10"))
PHASE_PLANNER_READ_TIMEOUT_SECONDS = float(os.getenv("PHASE_PLANNER_READ_TIMEOUT_SECONDS", "1800"))

# Character budgets for what actually gets folded into a single planning
# prompt. Kept well under the model's context window (roughly 4 chars/token)
# so there is real headroom left for the system prompt AND the model's own
# output -- a multi-phase plan for a large rewrite can itself run to several
# thousand tokens, and on CPU-only hardware output generation is the
# dominant cost, not input processing. These two budgets are a shared
# allowance, not each independently allowed the full window: the request
# text and the project audit are both folded into the SAME prompt, so both
# must be capped or one alone (previously the audit, which was never capped
# here) can push the combined prompt to the edge of the context window on
# its own. Set to use a meaningful share of PHASE_PLANNER_NUM_CTX below
# without maxing it out -- a bigger prompt means slower prefill, and this
# call already runs on a generous timeout rather than a tight one, so there
# is no need to be as conservative as when the window was still 8192.
PHASE_PLANNER_MAX_REQUEST_CONTEXT_CHARS = int(os.getenv("PHASE_PLANNER_MAX_REQUEST_CONTEXT_CHARS", "24000"))
PHASE_PLANNER_MAX_AUDIT_CONTEXT_CHARS = int(os.getenv("PHASE_PLANNER_MAX_AUDIT_CONTEXT_CHARS", "16000"))

# Sent explicitly on every phase-planner call instead of relying silently on
# whatever the Modelfile happens to bake in -- matches AegisCoder's original
# design principle ("num_ctx set explicitly... on every API call, never left
# at a silent default").
#
# IMPORTANT: this was raised from 8192 to 32768 on 2026-08-27 to match a
# deliberate fix made to the actual Modelfile (AegisCoder/AI/Modelfile.local-
# code-7b) on 2026-08-25, after multi-hundred-file project audits were
# silently exceeding the original 8192 window. If this call requested
# num_ctx=8192 while the Modelfile itself is baked for 32768, Ollama honors
# the per-call value -- so an unmatched default here would have silently
# UNDONE that fix for this specific call, shrinking it right back down to
# the window that caused the original problem. This value was NOT
# re-verified live against the Modelfile at the time of this change (the
# device bridge was unreachable) -- confirm with `ollama show local-code:7b
# --modelfile` and correct via .env if it has since changed.
PHASE_PLANNER_NUM_CTX = int(os.getenv("PHASE_PLANNER_NUM_CTX", "32768"))

# ---------------------------------------------------------------------------
# GroupChat safety limits
# ---------------------------------------------------------------------------
# Hard ceiling on total conversation turns. This is the direct replacement
# for AegisCoder's "unattended runtime budget" concept, scoped to a single
# task instead of a whole session: a dev/security disagreement can bounce
# back and forth but can never loop forever unattended.
MAX_GROUPCHAT_ROUNDS = int(os.getenv("MAX_GROUPCHAT_ROUNDS", "12"))

# A round limit is a context-window guard, not permission to ship unverified
# code. When a phase exhausts its conversation, start a fresh repair cycle.
MAX_PHASE_CYCLES = int(os.getenv("MAX_PHASE_CYCLES", "3"))

# ---------------------------------------------------------------------------
# Tool timeouts
# ---------------------------------------------------------------------------
SHELL_TIMEOUT_SECONDS = int(os.getenv("SHELL_TIMEOUT_SECONDS", "30"))
TEST_TIMEOUT_SECONDS = int(os.getenv("TEST_TIMEOUT_SECONDS", "120"))

# ---------------------------------------------------------------------------
# Large-deletion guard threshold -- see safety/deletion_guard.py
# ---------------------------------------------------------------------------
DELETION_GUARD_THRESHOLD = float(os.getenv("DELETION_GUARD_THRESHOLD", "0.40"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOG_DIR = os.getenv("LOG_DIR", "logs")