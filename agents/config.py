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

LLM_CONFIG: dict[str, Any] = {
    "config_list": [
        {
            "model": MODEL_NAME,
            "base_url": OLLAMA_OPENAI_BASE,
            "api_key": "ollama",  # unused by Ollama but required by the OpenAI client shape
        }
    ],
    "temperature": 0.2,
    "cache_seed": None,  # disable AutoGen's response cache -- always want fresh reasoning
}

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
