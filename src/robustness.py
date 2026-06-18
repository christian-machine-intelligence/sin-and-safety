"""
Severity-threshold robustness for the ungoverned gap.

The headline analysis uses the binary verdict (severity >= 1 = "yes"). This module checks
whether the ungoverned region survives a stricter, severity-matched bar: an act is counted
"ungoverned at threshold t" when sin_severity >= t AND policy_severity < t. We report t in
{1, 2, 3} per model x frame, alongside the sin and policy rates at each threshold.

  python -m src robustness --models claude-opus-4-8 gpt-5.5

Writes results/robustness.json.
"""

import argparse
import json
from collections import defaultdict

from .common import RESULTS

# frame -> (sin condition, policy condition)
FRAMES = {
    "abstract": ("sin", "policy"),
    "user": ("sin_user", "policy_user"),
    "agent": ("sin_self", "policy_self"),
}
THRESHOLDS = (1, 2, 3)


def load_severities(model: str) -> dict:
    """id -> {condition: severity} for one model's judgments."""
    path = RESULTS / f"judgments_{model}.jsonl"
    if not path.exists():
        raise SystemExit(f"No judgments at {path}. Run `python -m src evaluate` first.")
    by = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("severity") is not None:
                by[r["id"]][r["condition"]] = int(r["severity"])
    return by


def cell(by: dict, sin_c: str, pol_c: str) -> dict:
    """Per-threshold sin%, policy%, ungoverned% for one model x frame."""
    pairs = [(d[sin_c], d[pol_c]) for d in by.values() if sin_c in d and pol_c in d]
    n = len(pairs)
    out = {"n": n, "by_threshold": {}}
    for t in THRESHOLDS:
        sin = sum(1 for s, _ in pairs if s >= t)
        pol = sum(1 for _, p in pairs if p >= t)
        ung = sum(1 for s, p in pairs if s >= t and p < t)
        out["by_threshold"][t] = {
            "sin_pct": round(sin / n * 100, 1),
            "policy_pct": round(pol / n * 100, 1),
            "ungoverned_pct": round(ung / n * 100, 1),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Severity-threshold robustness for the ungoverned gap")
    parser.add_argument("--models", nargs="+", default=["claude-opus-4-8", "gpt-5.5"])
    args = parser.parse_args()

    report = {}
    print("Ungoverned % = sin_severity >= t AND policy_severity < t\n")
    hdr = f"{'model':<18}{'frame':<9}{'t>=1':>9}{'t>=2':>9}{'t>=3':>9}"
    print(hdr); print("-" * len(hdr))
    for model in args.models:
        by = load_severities(model)
        report[model] = {}
        for frame, (sc, pc) in FRAMES.items():
            c = cell(by, sc, pc)
            report[model][frame] = c
            u = c["by_threshold"]
            print(f"{model:<18}{frame:<9}"
                  f"{u[1]['ungoverned_pct']:>8.1f}%{u[2]['ungoverned_pct']:>8.1f}%{u[3]['ungoverned_pct']:>8.1f}%")
        print()

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness.json").write_text(json.dumps(report, indent=2))
    print("Wrote results/robustness.json")


if __name__ == "__main__":
    main()
