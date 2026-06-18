"""
Phase 2: Multi-condition judgment with Claude.

Every act is judged under a set of CONDITIONS, each an independent API call (no shared
context). A condition is (subject x axis):

  subject  abstract | user | self   (who performs the act)
  axis     sin | policy

  conditions: sin, policy                     (abstract baseline — the original run)
              sin_user, policy_user           ("if a user engaged in this act ...")
              sin_self, policy_self           ("if you engaged in this act ...")

Each judgment returns {"severity": 0-5, "binary": "yes|no", "rationale": "..."}.
Responses are cached by sha256(model | condition | act_id | prompt_version); the abstract
"sin"/"policy" keys match the original run, so that baseline is reused for free.

Output: results/judgments_<model>.jsonl, one row per (act, condition).
"""

import argparse
import asyncio
import hashlib
import json
import os
import time

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from .common import (
    BENCHMARK, CACHE, CONDITIONS, PROMPT_VERSION, RESULTS, extract_json, load_prompt, read_jsonl,
)

DEFAULT_MODEL = "claude-opus-4-8"
ALL_CONDITIONS = list(CONDITIONS.keys())
MAX_CONCURRENT = 20
MAX_RETRIES = 3
DEFAULT_EFFORT = "low"  # reasoning effort for OpenAI reasoning models (kept low for parity
                        # with the direct, non-thinking Anthropic judgments)


def provider_for(model: str) -> str:
    return "openai" if model.startswith("gpt") else "anthropic"


def cache_key(model: str, condition: str, act_id: str, act_text: str, effort: str) -> str:
    # Hash the ACT TEXT (not just the id) so reused ids across benchmark/pilot files can't
    # collide, and so editing an act busts its cache. Effort is namespaced for OpenAI
    # reasoning models (Claude judgments are direct, so effort is "na").
    act_h = hashlib.sha256(act_text.encode()).hexdigest()[:10]
    eff = effort if provider_for(model) == "openai" else "na"
    raw = f"{model}|{condition}|{act_id}|{act_h}|{eff}|{PROMPT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def cache_path(model: str, condition: str, act_id: str, act_text: str, effort: str):
    return CACHE / f"{cache_key(model, condition, act_id, act_text, effort)}.json"


def parse_judgment(text: str) -> dict | None:
    """Validate a model response into {severity:int 0-5, binary:'yes'/'no', rationale:str}."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return None
    try:
        sev = int(obj["severity"])
    except (KeyError, ValueError, TypeError):
        return None
    if not 0 <= sev <= 5:
        return None
    binary = str(obj.get("binary", "")).strip().lower()
    if binary not in ("yes", "no"):
        binary = "yes" if sev >= 1 else "no"
    return {
        "severity": sev,
        "binary": binary,
        "rationale": str(obj.get("rationale", "")).strip(),
    }


def _condition_meta(condition: str) -> dict:
    meta = CONDITIONS[condition]
    return {"condition": condition, "subject": meta["subject"], "axis": meta["axis"]}


async def _complete(model: str, prompt: str) -> str:
    """Single-turn completion dispatched to the right provider; returns the text."""
    if _runner["provider"] == "openai":
        resp = await _runner["client"].responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            reasoning={"effort": _runner["effort"]},
            max_output_tokens=2500,
        )
        text = getattr(resp, "output_text", "") or ""
        if not text:  # fall back to manual extraction
            for item in resp.output:
                if item.type == "message":
                    for part in item.content:
                        if getattr(part, "type", None) == "output_text":
                            text += part.text
        return text.strip()
    # Anthropic. NB: temperature is deprecated for Opus 4.8+ and must not be sent.
    resp = await _runner["client"].messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


async def judge_one(
    model: str,
    act: dict,
    condition: str,
    templates: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> dict:
    """Judge one (act, condition), using the cache when available."""
    meta = _condition_meta(condition)
    cpath = cache_path(model, condition, act["id"], act["act"], _runner["effort"])
    if cpath.exists():
        cached = json.loads(cpath.read_text())
        # Upgrade older cache rows (which only carried `axis`) with subject/condition.
        return {**cached, **meta, "cached": True}

    prompt = templates[condition].format(act=act["act"])
    parsed = None
    raw = None
    error = None
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                raw = await _complete(model, prompt)
                parsed = parse_judgment(raw)
                if parsed is not None:
                    break
            except Exception as e:  # noqa: BLE001
                error = str(e)
                await asyncio.sleep(1.5 * (attempt + 1))

    record = {
        "id": act["id"],
        "sin": act["sin"],
        **meta,
        "model": model,
        "severity": parsed["severity"] if parsed else None,
        "binary": parsed["binary"] if parsed else None,
        "rationale": parsed["rationale"] if parsed else None,
        "raw": raw,
        "error": error if parsed is None else None,
        # carry covariates through for analysis convenience
        "severity_intended": act.get("severity_intended"),
        "harm_target": act.get("harm_target"),
        "scope": act.get("scope"),
        "legality": act.get("legality"),
        "act": act["act"],
    }
    # Only cache successful parses (so failures get retried on the next run).
    if parsed is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(record, ensure_ascii=False))
    return {**record, "cached": False}


async def run_eval(model: str, acts: list[dict], conditions: list[str], concurrency: int):
    # template stem == condition name (e.g. data/prompts/policy_self_prompt.txt)
    templates = {c: load_prompt(c) for c in conditions}
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        judge_one(model, act, c, templates, semaphore)
        for act in acts
        for c in conditions
    ]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Judging ({model})")
    return results


# Active runner: {provider, client, effort}. Set via init_runner().
_runner: dict = {"provider": None, "client": None, "effort": DEFAULT_EFFORT}


def init_runner(model: str, effort: str = DEFAULT_EFFORT) -> None:
    provider = provider_for(model)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set (copy an existing study's .env).")
        _runner.update(provider="openai",
                       client=AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]), effort=effort)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set (copy an existing study's .env).")
        _runner.update(provider="anthropic",
                       client=AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]), effort=effort)


async def async_main():
    parser = argparse.ArgumentParser(description="Multi-condition sin/policy judgment (Claude or GPT)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS + ["all"], default=["all"],
                        help="which (subject x axis) conditions to judge")
    parser.add_argument("--limit", type=int, default=None, help="judge only the first N acts")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="reasoning effort for OpenAI reasoning models (ignored for Claude)")
    parser.add_argument("--benchmark", default=None, help="override acts path (e.g. pilot file)")
    parser.add_argument("--out", default=None, help="override output jsonl path")
    args = parser.parse_args()

    from pathlib import Path
    bench_path = Path(args.benchmark) if args.benchmark else BENCHMARK
    if not bench_path.exists():
        raise SystemExit(f"Benchmark not found: {bench_path}. Run `python -m src generate` first.")

    acts = read_jsonl(bench_path)
    if args.limit:
        acts = acts[: args.limit]
    conditions = ALL_CONDITIONS if "all" in args.conditions else args.conditions

    init_runner(args.model, args.effort)

    start = time.time()
    results = await run_eval(args.model, acts, conditions, args.concurrency)
    elapsed = time.time() - start

    out_path = Path(args.out) if args.out else RESULTS / f"judgments_{args.model}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist clean rows (drop the transient 'cached' flag).
    with open(out_path, "w") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k != "cached"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_cached = sum(1 for r in results if r.get("cached"))
    n_fail = sum(1 for r in results if r.get("severity") is None)
    print(f"\nWrote {len(results)} judgments ({len(conditions)} conditions x {len(acts)} acts) "
          f"to {out_path} in {elapsed/60:.1f} min.")
    print(f"  cache hits: {n_cached}/{len(results)}   parse/API failures: {n_fail}")
    if n_fail:
        print("  (failures are NOT cached; re-run to retry them.)")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
