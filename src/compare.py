"""
Cross-model comparison.

Reads results/summary_stats_<model>.json for two or more models and emits:
  - a console table of sin/policy yes-rates, gap rate, and severities by (model, subject)
  - results/figures/crossmodel_gap.png  (gap rate by model x subject)
  - results/figures/crossmodel_axes.png (sin & policy yes-rate by model, abstract framing)
  - results/crossmodel_summary.json

  python -m src compare --models claude-opus-4-8 gpt-5.5
"""

import argparse
import json
import re

from .common import FIGURES, RESULTS, SUBJECTS


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def load_summary(model: str) -> dict:
    path = RESULTS / f"summary_stats_{_slug(model)}.json"
    if not path.exists():
        # fall back to un-slugged name for older runs
        alt = RESULTS / f"summary_stats_{model}.json"
        path = alt if alt.exists() else path
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python -m src analyze --model {model}` first.")
    return json.loads(path.read_text())


def make_figures(models: list[str], summaries: dict[str, dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES.mkdir(parents=True, exist_ok=True)
    subjects = SUBJECTS

    # Gap rate by model x subject.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(subjects))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        bs = summaries[m]["by_subject"]
        vals = [bs.get(s, {}).get("gap_quadrant_rate", 0) or 0 for s in subjects]
        bars = ax.bar(x + i * w, vals, w, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(subjects)
    ax.set_ylim(0, 1); ax.set_ylabel("gap rate (sinful, not policy-condemned)")
    ax.set_title("Sin/policy gap by model and acting subject"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "crossmodel_gap.png", dpi=150); plt.close(fig)

    # Sin & policy yes-rate by model (abstract framing).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)  # sin, policy
    for i, m in enumerate(models):
        b = summaries[m]["by_subject"].get("abstract", {})
        vals = [b.get("sin_yes_rate", 0), b.get("policy_yes_rate", 0)]
        bars = ax.bar(x + i * w, vals, w, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(["sinful", "policy-condemned"])
    ax.set_ylim(0, 1); ax.set_ylabel("fraction judged 'yes' (abstract framing)")
    ax.set_title("Sin vs policy yes-rate by model"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "crossmodel_axes.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare sin/policy results across models")
    parser.add_argument("--models", nargs="+", default=["claude-opus-4-8", "gpt-5.5"])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    summaries = {m: load_summary(m) for m in args.models}

    # Console table.
    print(f"\n=== Cross-model comparison ===")
    hdr = f"{'model':<20}{'subject':<10}{'sin%':>7}{'policy%':>9}{'gap%':>7}{'sinSev':>8}{'polSev':>8}"
    print(hdr); print("-" * len(hdr))
    out = {}
    for m in args.models:
        out[m] = {}
        for s in SUBJECTS:
            b = summaries[m]["by_subject"].get(s)
            if not b:
                continue
            out[m][s] = {k: b[k] for k in
                         ["sin_yes_rate", "policy_yes_rate", "gap_quadrant_rate",
                          "mean_sin_severity", "mean_policy_severity"]}
            print(f"{m:<20}{s:<10}{b['sin_yes_rate']*100:>6.0f}%{b['policy_yes_rate']*100:>8.0f}%"
                  f"{b['gap_quadrant_rate']*100:>6.0f}%{b['mean_sin_severity']:>8.2f}"
                  f"{b['mean_policy_severity']:>8.2f}")

    # Self-exemption (can the model sin?) side by side.
    print("\nSelf-exemption (of acts sinful for a user, fraction the model exempts itself from):")
    for m in args.models:
        se = summaries[m].get("subject_comparison", {}).get("self_exemption", {})
        if se:
            print(f"  {m:<20} rate={se.get('self_exemption_rate')}  "
                  f"category-error rationale rate={se.get('self_sin_category_error_rate')}")

    (RESULTS / "crossmodel_summary.json").write_text(json.dumps(out, indent=2))
    if not args.no_figures:
        make_figures(args.models, summaries)
    print("\nWrote results/crossmodel_summary.json"
          + ("" if args.no_figures else " and results/figures/crossmodel_*.png"))


if __name__ == "__main__":
    main()
