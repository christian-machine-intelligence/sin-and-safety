"""Shared helpers: paths, env loading, JSON extraction, the seven capital sins."""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROMPTS = DATA / "prompts"
BENCHMARK = DATA / "benchmark" / "acts.jsonl"
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
FIGURES = RESULTS / "figures"

# Versioned so changing a prompt template busts the cache cleanly.
PROMPT_VERSION = "v1"

# The seven capital sins (Latin in parentheses for reference).
SINS = ["pride", "greed", "lust", "envy", "gluttony", "wrath", "sloth"]
SIN_LATIN = {
    "pride": "superbia", "greed": "avaritia", "lust": "luxuria", "envy": "invidia",
    "gluttony": "gula", "wrath": "ira", "sloth": "acedia",
}

# Allowed covariate values (used for benchmark validation).
SEVERITY_LEVELS = ["mild", "moderate", "grave"]
HARM_TARGETS = ["none", "self", "others"]
SCOPES = ["private", "public"]
LEGALITY = ["legal", "illegal", "ambiguous"]

ACTS_PER_SIN = 100

# Judgment conditions: subject (who performs the act) x axis (sin / policy).
# The condition NAME doubles as the cache-key token and the prompt-template stem.
# "sin"/"policy" deliberately match the original abstract-framing run so that the
# already-collected baseline judgments stay cache-valid and do not re-bill.
CONDITIONS = {
    "sin":         {"subject": "abstract", "axis": "sin"},
    "policy":      {"subject": "abstract", "axis": "policy"},
    "sin_user":    {"subject": "user",     "axis": "sin"},
    "policy_user": {"subject": "user",     "axis": "policy"},
    "sin_self":    {"subject": "self",     "axis": "sin"},
    "policy_self": {"subject": "self",     "axis": "policy"},
}
SUBJECTS = ["abstract", "user", "self"]

# Load .env from the study root once on import.
load_dotenv(ROOT / ".env")


def load_prompt(name: str) -> str:
    """Load a prompt template by short name ('sin' or 'policy')."""
    return (PROMPTS / f"{name}_prompt.txt").read_text()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response, tolerantly.

    Handles bare JSON, ```json fenced blocks, and leading/trailing prose.
    Returns None if no parseable object is found.
    """
    if not text:
        return None
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # Fall back to the first balanced-looking {...} span.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
