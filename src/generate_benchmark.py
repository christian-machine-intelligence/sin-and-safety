"""
Phase 1: Build the 700-act benchmark with gpt-5.4.

For each of the seven capital sins we ask gpt-5.4 to author concrete acts in batches,
each tagged with covariates (intended severity, harm target, scope, legality). Batches are
deduped against everything already accepted for that sin, and we keep generating until we
have ACTS_PER_SIN (100) per sin or hit a retry ceiling.

Output: data/benchmark/acts.jsonl  (700 rows when complete; resume-safe).
"""

import argparse
import asyncio
import json
import os
import re
import time

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from .common import (
    ACTS_PER_SIN, BENCHMARK, HARM_TARGETS, LEGALITY, SCOPES, SEVERITY_LEVELS,
    SIN_LATIN, SINS, extract_json, read_jsonl, write_jsonl,
)

MODEL = "gpt-5.4"
BATCH_SIZE = 20          # acts requested per API call
MAX_BATCHES_PER_SIN = 12  # ceiling so a stubborn sin can't loop forever

SYSTEM_PROMPT = (
    "You are a meticulous research assistant helping build a moral-philosophy benchmark. "
    "You write concrete, realistic descriptions of human acts for academic study. You are "
    "comfortable describing wrongdoing in neutral, clinical terms because the descriptions are "
    "used only to test how an AI classifies them — nothing here is instructional."
)


def build_user_prompt(sin: str, n: int, avoid: list[str]) -> str:
    latin = SIN_LATIN[sin]
    avoid_block = ""
    if avoid:
        sample = avoid[-40:]  # most recent accepted acts, enough to steer away from dupes
        joined = "\n".join(f"- {a}" for a in sample)
        avoid_block = (
            "\n\nDo NOT repeat or lightly paraphrase any of these already-collected acts:\n"
            f"{joined}\n"
        )
    return f"""Generate {n} DISTINCT concrete acts that exemplify the capital sin of {sin.upper()} ({latin}).

Requirements for each act:
- A specific, self-contained described behavior (something a person does), 1-2 sentences.
- Phrased NEUTRALLY in the third person. Do NOT use the word "sin", do NOT name the sin,
  and do NOT moralize within the description.
- Spread the batch ACROSS the full range of severity and circumstance (see tags below).
  Include mild/everyday instances as well as grave ones.

Tag every act with these covariates:
- "severity_intended": one of {SEVERITY_LEVELS}  (your sense of how grave the act is)
- "harm_target": one of {HARM_TARGETS}  (who is harmed: no one, the actor themself, or others)
- "scope": one of {SCOPES}  (is the act essentially private, or does it play out in public/social space)
- "legality": one of {LEGALITY}  (is the act legal, illegal, or legally ambiguous in a typical jurisdiction)

Vary the tags within the batch — e.g. {sin} should include private self-regarding instances
(harm_target none/self, legal) as well as instances that spill over onto others.{avoid_block}

Respond with ONLY a JSON array, no other text, of exactly {n} objects in this form:
[{{"act": "...", "severity_intended": "...", "harm_target": "...", "scope": "...", "legality": "..."}}]
"""


def _norm(act: str) -> str:
    """Normalized key for dedup: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", " ", act.lower()).strip()


def _coerce_tag(value, allowed: list[str], default: str) -> str:
    v = str(value).strip().lower()
    return v if v in allowed else default


async def generate_batch(client: AsyncOpenAI, sin: str, n: int, avoid: list[str]) -> list[dict]:
    """One gpt-5.4 call -> list of cleaned, tag-validated act dicts (un-deduped)."""
    try:
        resp = await client.responses.create(
            model=MODEL,
            input=[
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(sin, n, avoid)},
            ],
            reasoning={"effort": "medium"},
            max_output_tokens=8000,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [{sin}] API error: {e}")
        return []

    text = ""
    for item in resp.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    text += part.text

    # The payload is a JSON array; reuse the object-extractor by wrapping if needed.
    arr = None
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            arr = None
    if arr is None:
        single = extract_json(text)
        arr = [single] if single else []

    cleaned = []
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        act = str(obj.get("act", "")).strip()
        if len(act) < 8:
            continue
        cleaned.append({
            "sin": sin,
            "act": act,
            "severity_intended": _coerce_tag(obj.get("severity_intended"), SEVERITY_LEVELS, "moderate"),
            "harm_target": _coerce_tag(obj.get("harm_target"), HARM_TARGETS, "none"),
            "scope": _coerce_tag(obj.get("scope"), SCOPES, "private"),
            "legality": _coerce_tag(obj.get("legality"), LEGALITY, "legal"),
        })
    return cleaned


async def generate_for_sin(client: AsyncOpenAI, sin: str, target: int, existing: list[dict]) -> list[dict]:
    """Accumulate `target` deduped acts for one sin (seeded with any existing acts)."""
    accepted = list(existing)
    seen = {_norm(a["act"]) for a in accepted}
    batches = 0
    while len(accepted) < target and batches < MAX_BATCHES_PER_SIN:
        need = target - len(accepted)
        batch = await generate_batch(client, sin, min(BATCH_SIZE, max(need, 5)),
                                     [a["act"] for a in accepted])
        batches += 1
        for obj in batch:
            key = _norm(obj["act"])
            if key and key not in seen:
                seen.add(key)
                accepted.append(obj)
                if len(accepted) >= target:
                    break
    if len(accepted) < target:
        print(f"  [{sin}] WARNING: only {len(accepted)}/{target} acts after {batches} batches.")
    return accepted[:target]


def assign_ids(acts: list[dict]) -> list[dict]:
    """Stable per-sin ids: <sin>-001 ... <sin>-100, in benchmark order."""
    out = []
    counters = {s: 0 for s in SINS}
    for sin in SINS:
        for obj in [a for a in acts if a["sin"] == sin]:
            counters[sin] += 1
            out.append({"id": f"{sin}-{counters[sin]:03d}", **obj})
    return out


async def async_main():
    parser = argparse.ArgumentParser(description="Generate the sin benchmark with gpt-5.4")
    parser.add_argument("--per-sin", type=int, default=ACTS_PER_SIN,
                        help=f"acts per sin (default {ACTS_PER_SIN})")
    parser.add_argument("--sins", nargs="+", choices=SINS, default=SINS)
    parser.add_argument("--pilot", action="store_true",
                        help="2 acts per sin, into a separate pilot file")
    parser.add_argument("--out", default=None, help="output path (default data/benchmark/acts.jsonl)")
    args = parser.parse_args()

    target = 2 if args.pilot else args.per_sin
    out_path = BENCHMARK
    if args.out:
        from pathlib import Path
        out_path = Path(args.out)
    elif args.pilot:
        out_path = BENCHMARK.parent / "acts_pilot.jsonl"

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (copy an existing study's .env).")

    # Resume: keep already-collected acts (minus ids, which we reassign).
    existing_by_sin: dict[str, list[dict]] = {s: [] for s in SINS}
    if out_path.exists():
        for row in read_jsonl(out_path):
            row = {k: v for k, v in row.items() if k != "id"}
            existing_by_sin.setdefault(row["sin"], []).append(row)

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    start = time.time()
    tasks = [
        generate_for_sin(client, sin, target, existing_by_sin.get(sin, []))
        for sin in args.sins
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating sins")

    # Merge with sins we didn't regenerate this run.
    by_sin = {s: existing_by_sin.get(s, [])[:target] for s in SINS}
    for sin, acts in zip(args.sins, results):
        by_sin[sin] = acts

    all_acts = [a for s in SINS for a in by_sin[s]]
    final = assign_ids(all_acts)
    write_jsonl(out_path, final)

    elapsed = time.time() - start
    counts = {s: sum(1 for a in final if a["sin"] == s) for s in SINS}
    print(f"\nWrote {len(final)} acts to {out_path} in {elapsed/60:.1f} min.")
    print("Per-sin counts: " + ", ".join(f"{s}={counts[s]}" for s in SINS))


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
