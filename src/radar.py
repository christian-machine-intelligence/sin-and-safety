"""
Radar-chart artifacts: sin vs policy ("secular") per capital sin, by model x subject.

Renders two 2-row (model) x 3-column (subject) grids of radar plots into results/figures/:
  - radar_binary_by_subject.png    (fraction judged yes, 0-100%)
  - radar_severity_by_subject.png  (mean Likert severity, 0-5)

Reads results/summary_stats_<slug>.json (produced by `python -m src analyze`).

  python -m src radar --models claude-opus-4-8 gpt-5.5
"""

import argparse
import json
import re

import numpy as np

from .common import FIGURES, RESULTS, SINS, SUBJECTS

SIN_COLOR = "#D85A30"
POL_COLOR = "#378ADD"
MODEL_LABELS = {"claude-opus-4-8": "Claude Opus 4.8", "gpt-5.5": "GPT-5.5"}
SUBJECT_LABELS = {"abstract": "abstract", "user": "user (“if a user…”)", "self": "self (“if you…”)"}


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def load_per_sin(model: str) -> dict:
    path = RESULTS / f"summary_stats_{_slug(model)}.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python -m src analyze --model {model}` first.")
    return json.loads(path.read_text())["by_subject"]


def series(per_sin: dict, metric: str) -> tuple[list[float], list[float]]:
    """Return (sin_values, policy_values) across SINS for the given metric."""
    sin_v, pol_v = [], []
    for s in SINS:
        ps = per_sin[s]
        if metric == "binary":
            sin_v.append((ps["sin_and_policy"] + ps["sin_not_policy"]) / ps["n"] * 100)
            pol_v.append((ps["sin_and_policy"] + ps["policy_not_sin"]) / ps["n"] * 100)
        else:  # severity
            sin_v.append(ps["mean_sin_severity"])
            pol_v.append(ps["mean_policy_severity"])
    return sin_v, pol_v


def make_grid(models: list[str], by_model: dict, metric: str, out_name: str, which: str = "both"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = 100 if metric == "binary" else 5
    ticks = [25, 50, 75, 100] if metric == "binary" else [1, 2, 3, 4, 5]
    fmt = (lambda v: f"{int(v)}%") if metric == "binary" else (lambda v: f"{int(v)}")
    subjects = SUBJECTS

    angles = np.linspace(0, 2 * np.pi, len(SINS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(len(models), len(subjects), figsize=(12, 8.2),
                             subplot_kw=dict(polar=True))
    if len(models) == 1:
        axes = np.array([axes])

    for r, model in enumerate(models):
        bs = by_model[model]
        for c, subj in enumerate(subjects):
            ax = axes[r][c]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(SINS, fontsize=9)
            ax.set_ylim(0, vmax)
            ax.set_yticks(ticks)
            ax.set_yticklabels([fmt(t) for t in ticks], fontsize=7, color="0.5")
            ax.grid(color="0.8", linewidth=0.6)

            sin_v, pol_v = series(bs[subj]["per_sin"], metric)
            layers = [(sin_v, SIN_COLOR)]
            if which == "both":
                layers.append((pol_v, POL_COLOR))
            for vals, color in layers:
                vv = vals + vals[:1]
                ax.plot(angles, vv, color=color, linewidth=2)
                ax.fill(angles, vv, color=color, alpha=0.17)
            if r == 0:
                ax.set_title(SUBJECT_LABELS.get(subj, subj), fontsize=12, pad=14)
            if c == 0:
                ax.text(-0.38, 0.5, MODEL_LABELS.get(model, model), rotation=90,
                        va="center", ha="center", fontsize=13, fontweight="medium",
                        transform=ax.transAxes)

    metric_name = "% judged yes" if metric == "binary" else "mean severity (0–5)"
    if which == "sin":
        title = f"Judged sinful (orange), per capital sin — {metric_name}"
    else:
        title = f"Sinful (orange) vs condemned by safety policy (blue) — {metric_name}"
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / out_name, dpi=150)
    plt.close(fig)
    print(f"  wrote results/figures/{out_name}")


def make_pair(models: list[str], by_model: dict, which: str, out_name: str, frame: str = "abstract"):
    """Clean single-frame card: the models side by side, binary %, for social sharing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    angles = np.linspace(0, 2 * np.pi, len(SINS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models) + 0.6, 5.9),
                             subplot_kw=dict(polar=True))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sin_v, pol_v = series(by_model[model][frame]["per_sin"], "binary")
        layers = [(sin_v, SIN_COLOR)]
        if which == "both":
            layers.append((pol_v, POL_COLOR))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(SINS, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100%"], fontsize=8, color="0.5")
        ax.grid(color="0.8", linewidth=0.6)
        for vals, color in layers:
            vv = vals + vals[:1]
            ax.plot(angles, vv, color=color, linewidth=2.6)
            ax.fill(angles, vv, color=color, alpha=0.18)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=16, fontweight="medium", pad=20)

    if which == "both":
        handles = [Line2D([0], [0], color=SIN_COLOR, lw=3, label="judged sinful"),
                   Line2D([0], [0], color=POL_COLOR, lw=3, label="condemned by safety policy")]
        title = "Sin vs. safety policy, per capital sin"
    else:
        handles = [Line2D([0], [0], color=SIN_COLOR, lw=3, label="judged sinful")]
        title = "What each model calls sin, per capital sin"
    fig.suptitle(title, fontsize=17, y=0.99)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.94])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / out_name, dpi=200)
    plt.close(fig)
    print(f"  wrote results/figures/{out_name}")


def main():
    parser = argparse.ArgumentParser(description="Render sin-vs-policy radar artifacts")
    parser.add_argument("--models", nargs="+", default=["claude-opus-4-8", "gpt-5.5"])
    args = parser.parse_args()
    by_model = {m: load_per_sin(m) for m in args.models}
    print("Rendering radar artifacts...")
    make_grid(args.models, by_model, "binary", "radar_sin_only_by_subject.png", which="sin")
    make_grid(args.models, by_model, "binary", "radar_binary_by_subject.png", which="both")
    make_grid(args.models, by_model, "severity", "radar_severity_by_subject.png")
    # Social-sharing cards: abstract frame, both models side by side.
    make_pair(args.models, by_model, "sin", "social_abstract_sin_pair.png")
    make_pair(args.models, by_model, "both", "social_abstract_overlay_pair.png")


if __name__ == "__main__":
    main()
