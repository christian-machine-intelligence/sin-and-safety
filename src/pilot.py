"""
End-to-end smoke test: 2 acts per sin (14 total), both phases, then analysis.

Confirms prompts render, gpt-5.4 generation + tagging works, Claude judgments parse,
the cache writes, and the analysis runs. Cheap enough to run before a full study.

  python -m src pilot [--model claude-opus-4-8]
"""

import argparse
import asyncio
import os

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from . import evaluate, generate_benchmark
from .analysis import contingency, load_judgments, pivot_subject
from .common import BENCHMARK, RESULTS, SINS, SUBJECTS, read_jsonl, write_jsonl

PILOT_ACTS = BENCHMARK.parent / "acts_pilot.jsonl"
PILOT_JUDGMENTS = RESULTS / "judgments_pilot.jsonl"
PER_SIN = 2


async def _run(model: str):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} not set (copy an existing study's .env).")

    # Phase 1 — 2 acts per sin.
    print(f"[pilot] generating {PER_SIN} acts/sin with gpt-5.4 ...")
    oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    existing = {s: [] for s in SINS}
    if PILOT_ACTS.exists():
        for row in read_jsonl(PILOT_ACTS):
            existing.setdefault(row["sin"], []).append({k: v for k, v in row.items() if k != "id"})
    per_sin_acts = await asyncio.gather(*[
        generate_benchmark.generate_for_sin(oai, s, PER_SIN, existing.get(s, [])) for s in SINS
    ])
    acts = generate_benchmark.assign_ids([a for lst in per_sin_acts for a in lst])
    write_jsonl(PILOT_ACTS, acts)
    print(f"[pilot] wrote {len(acts)} acts to {PILOT_ACTS}")

    # Phase 2 — judge all conditions (subject x axis).
    print(f"[pilot] judging all conditions with {model} ...")
    evaluate.init_runner(model)
    results = await evaluate.run_eval(model, acts, evaluate.ALL_CONDITIONS, concurrency=8)
    import json
    with open(PILOT_JUDGMENTS, "w") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "cached"}, ensure_ascii=False) + "\n")
    n_fail = sum(1 for r in results if r.get("severity") is None)
    print(f"[pilot] wrote {len(results)} judgments to {PILOT_JUDGMENTS} ({n_fail} failures)")

    # Phase 3 — quick analysis across subjects + show a few rationales.
    df = load_judgments(model, PILOT_JUDGMENTS)
    subjects = [s for s in SUBJECTS if s in set(df["subject"])]
    print(f"\n[pilot] contingency by subject (n acts each):")
    wides = {}
    for subj in subjects:
        w = pivot_subject(df, subj)
        wides[subj] = w
        print(f"  {subj:<9} n={len(w)}  {contingency(w)}")
    base = wides.get("abstract", wides[subjects[0]])
    print("[pilot] sample judgments (abstract framing):")
    for _, row in base.head(3).iterrows():
        print(f"  - [{row['sin']}] {row['act'][:70]}...")
        print(f"      sin: sev={row['sin_severity']} ({row['sin_yes']})   "
              f"policy: sev={row['policy_severity']} ({row['policy_yes']})")
    print("\n[pilot] OK — all conditions render, JSON parses, cache + analysis run.")


def main():
    parser = argparse.ArgumentParser(description="End-to-end pilot smoke test")
    parser.add_argument("--model", default=evaluate.DEFAULT_MODEL)
    args = parser.parse_args()
    asyncio.run(_run(args.model))


if __name__ == "__main__":
    main()
