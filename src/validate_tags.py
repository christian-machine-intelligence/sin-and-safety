"""
Covariate-tag validity audit.

The covariate regression rests on harm_target / scope / legality tags that gpt-5.4 self-assigned
during benchmark generation. Here an INDEPENDENT model (default Claude Opus 4.8) re-tags a random
sample of acts; we report per-dimension % agreement and Cohen's kappa against the original tags.

  python -m src validate-tags [--n 140] [--model claude-opus-4-8]

Writes results/tag_audit.json.
"""

import argparse
import asyncio
import json
import os
import random

from anthropic import AsyncAnthropic

from .common import HARM_TARGETS, LEGALITY, RESULTS, SCOPES, SINS, extract_json, read_jsonl, BENCHMARK

DIMS = {"harm_target": HARM_TARGETS, "scope": SCOPES, "legality": LEGALITY}

PROMPT = """Classify the following act on three dimensions. Judge the act as described, neutrally.

Act: "{act}"

- harm_target: who, if anyone, is harmed — one of {harm}
    ("none" = victimless, "self" = harms mainly the actor, "others" = harms other people)
- scope: is the act essentially {scope} ("private" = personal/solitary, "public" = plays out socially)
- legality: in a typical jurisdiction, is it one of {legality}

Respond with ONLY a JSON object: {{"harm_target": "...", "scope": "...", "legality": "..."}}"""


def cohens_kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    n = len(a)
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        if x in idx and y in idx:
            obs[idx[x]][idx[y]] += 1
    po = sum(obs[i][i] for i in range(k)) / n
    ra = [sum(obs[i]) / n for i in range(k)]
    cb = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(ra[i] * cb[i] for i in range(k))
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


async def retag(client, model, act, sem):
    prompt = PROMPT.format(act=act["act"], harm=HARM_TARGETS, scope=SCOPES, legality=LEGALITY)
    async with sem:
        for _ in range(3):
            try:
                resp = await client.messages.create(
                    model=model, max_tokens=512,
                    messages=[{"role": "user", "content": prompt}])
                raw = "".join(b.text for b in resp.content if b.type == "text")
                obj = extract_json(raw)
                if isinstance(obj, dict) and all(d in obj for d in DIMS):
                    return {d: str(obj[d]).strip().lower() for d in DIMS}
            except Exception:  # noqa: BLE001
                await asyncio.sleep(1.5)
    return None


async def _run(n, model):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set.")
    acts = read_jsonl(BENCHMARK)
    rng = random.Random(42)
    # stratified: even sample per sin
    per = max(1, n // len(SINS))
    sample = []
    for s in SINS:
        pool = [a for a in acts if a["sin"] == s]
        sample += rng.sample(pool, min(per, len(pool)))
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(15)
    preds = await asyncio.gather(*[retag(client, model, a, sem) for a in sample])

    orig = {d: [] for d in DIMS}
    new = {d: [] for d in DIMS}
    ok = 0
    for a, p in zip(sample, preds):
        if p is None:
            continue
        ok += 1
        for d in DIMS:
            orig[d].append(a[d]); new[d].append(p[d])

    report = {"auditor_model": model, "n_sampled": len(sample), "n_scored": ok, "dimensions": {}}
    print(f"\n=== Covariate-tag audit: {model} vs gpt-5.4 tags (n={ok}) ===")
    print(f"{'dimension':<14}{'agreement':>12}{'kappa':>10}")
    for d in DIMS:
        agree = sum(x == y for x, y in zip(orig[d], new[d])) / len(orig[d])
        kap = cohens_kappa(orig[d], new[d], DIMS[d])
        report["dimensions"][d] = {"agreement": round(agree, 3), "cohens_kappa": round(kap, 3)}
        print(f"{d:<14}{agree*100:>11.0f}%{kap:>10.2f}")
    (RESULTS / "tag_audit.json").write_text(json.dumps(report, indent=2))
    print("\nWrote results/tag_audit.json")


def main():
    parser = argparse.ArgumentParser(description="Audit benchmark covariate tags vs an independent model")
    parser.add_argument("--n", type=int, default=140)
    parser.add_argument("--model", default="claude-opus-4-8")
    args = parser.parse_args()
    asyncio.run(_run(args.n, args.model))


if __name__ == "__main__":
    main()
