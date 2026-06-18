# Sin and Safety

<p align="center">
  <img src="banner.jpg" width="600" alt="Hieronymus Bosch, The Seven Deadly Sins and the Four Last Things (c. 1500). Oil on panel, Museo del Prado, Madrid. Public domain.">
</p>

<p align="center"><em>Hieronymus Bosch, "The Seven Deadly Sins and the Four Last Things" (c. 1500), Museo del Prado. The eye of God encircles the seven vices; the iris reads</em> Cave Cave Deus Videt <em>— "Beware, beware, God sees."</em></p>

Code and data for **ICMI Working Paper No. 25**, *"Cleanse Thou Me from Secret Faults: Initial Explorations on Sin and Agentic Alignment."*

A frontier model holds two moral self-understandings: a broad notion of **sin** absorbed from pretraining, and its understanding of the **safety policy** that governs it. We lay them over each other. Each of 700 acts — 100 under each of the seven capital vices — is judged twice, in independent contexts: *is this sinful?* and *is this disallowed by your safety policy?* **The ungoverned sins** are the divergence: acts a model recognizes as sinful but understands its own policy not to reach. They fall along a *harm* line, concentrating in the private, self-regarding vices (gluttony, sloth, pride) that a harm-based policy leaves untouched — the capital vices that, for an *agent* acting in the world, are the generative roots of real harm. (Both axes are the model's *self-reported understandings*, not its real operative policy or its deployed behavior.)

This repository is built for full reproducibility: the benchmark, every model judgment (with raw responses), the derived statistics, and the figures are all regenerable from the code with two API keys.

## Reproducing the study

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in OPENAI_API_KEY + ANTHROPIC_API_KEY
```

The full pipeline (`python -m src <command>`):

```bash
# Phase 1 — build the 700-act benchmark with gpt-5.4
python -m src generate                                   # -> data/benchmark/acts.jsonl

# Phase 2 — two-axis judgments under all six conditions, per model
python -m src evaluate --model claude-opus-4-8 --conditions all
python -m src evaluate --model gpt-5.5 --conditions all --effort none

# Phase 3 — per-model analysis (stats + figures)
python -m src analyze --model claude-opus-4-8 --raw
python -m src analyze --model gpt-5.5 --raw

# Phase 4 — cross-model comparison and the radar figures used in the paper
python -m src compare --models claude-opus-4-8 gpt-5.5
python -m src radar   --models claude-opus-4-8 gpt-5.5

# Robustness — ungoverned gap at stricter severity thresholds (paper Table 2)
python -m src robustness --models claude-opus-4-8 gpt-5.5

# Validity — independent re-tagging of the covariates (Cohen's kappa)
python -m src validate-tags --n 140

# Optional — cheap end-to-end smoke test (14 acts, all conditions)
python -m src pilot
```

**Determinism & cost.** Judgments are issued one (act × condition) per call and **cached by a hash of the act text** under `results/cache/` (gitignored), so re-runs cost nothing and reproduce byte-for-byte. Anthropic's `temperature` is deprecated on Opus 4.8, so Claude judgments are taken directly; GPT-5.5 is run at reasoning effort `none` for parity with Claude's non-thinking, direct judgments. A from-scratch run is ~4,200 judgments per model (≈$25 for Claude, ≈$12 for GPT-5.5 at the time of writing). Benchmark generation is ~$3–5.

**To regenerate only the paper's two figures** (sin-only and overlay) from the committed judgments, without any API calls:

```bash
python -m src analyze --model claude-opus-4-8 --raw
python -m src analyze --model gpt-5.5 --raw
python -m src radar   --models claude-opus-4-8 gpt-5.5
# -> results/figures/radar_sin_only_by_subject.png  (paper Figure 1)
# -> results/figures/radar_binary_by_subject.png    (paper Figure 2)
```

## Design

| | |
|---|---|
| **Benchmark** | 700 acts, 100 per capital sin (pride, greed, lust, envy, gluttony, wrath, sloth), authored by `gpt-5.4` |
| **Axes** | `sin` ("is this sinful?") and `policy` ("is this disallowed/condemned by your safety policy?") |
| **Frames** | `abstract` ("this act"), `user` ("if a user…"), `agent` ("if you…") |
| **Conditions** | the 6 crosses of frame × axis (`sin`, `policy`, `sin_user`, `policy_user`, `sin_self`, `policy_self`) |
| **Judges** | `claude-opus-4-8`, `gpt-5.5` |
| **Output** | 0–5 severity + binary per judgment; the paper's headline is the **binary**, with a severity-matched robustness check (Table 2); full Likert deferred to future work |

## Repository layout

```
src/
  generate_benchmark.py   # Phase 1 — author the benchmark (gpt-5.4)
  evaluate.py             # Phase 2 — multi-condition judgments (Claude or GPT), cached
  analysis.py             # Phase 3 — per-model contingency, McNemar, per-sin gap, figures
  compare.py              # Phase 4 — cross-model comparison table + figures
  radar.py                # radar figures (sin-only, overlay, severity)
  robustness.py           # severity-threshold robustness (paper Table 2)
  validate_tags.py        # covariate-tag audit (independent re-tag, Cohen's kappa)
  pilot.py                # end-to-end smoke test
  common.py               # paths, conditions registry, JSON helpers
data/
  benchmark/acts.jsonl    # the 700-act benchmark (input of record)
  prompts/*.txt           # the six judgment prompt templates
results/
  judgments_<model>.jsonl     # canonical output: one row per (act, condition), incl. raw response
  summary_stats_<model>.json  # per-model aggregates (contingency, McNemar, per-sin, regression)
  per_act_<model>_<frame>.jsonl  # per-act sin/policy join, per frame
  crossmodel_summary.json     # cross-model comparison
  robustness.json             # ungoverned gap at severity >= 1/2/3
  tag_audit.json              # covariate-validity kappas
  figures/                    # regenerated by analyze / compare / radar
  cache/                      # per-call response cache (gitignored, regenerable)
```

## Data dictionary

**`data/benchmark/acts.jsonl`** — one act per line:

```json
{"id": "gluttony-001", "sin": "gluttony", "act": "<behavior>",
 "severity_intended": "mild|moderate|grave", "harm_target": "none|self|others",
 "scope": "private|public", "legality": "legal|illegal|ambiguous"}
```

**`results/judgments_<model>.jsonl`** — one judgment per line (the data of record; `raw` retains the model's verbatim response):

```json
{"id": "gluttony-001", "sin": "gluttony", "condition": "policy_self",
 "subject": "self", "axis": "policy", "model": "gpt-5.5",
 "severity": 0, "binary": "no", "rationale": "...", "raw": "...", "error": null,
 "severity_intended": "...", "harm_target": "...", "scope": "...", "legality": "...", "act": "..."}
```

Covariate tags were validated by independent re-tagging (Cohen's *κ*: harm_target 0.79, scope 0.73, legality 0.73; see `results/tag_audit.json`).

## Notes

- `RESEARCH_PLAN.md` is the original pre-registration; the final study extended it to two judge models and the three subject frames (see the paper).
- `.env` is required to run Phases 1–2 and is gitignored — never commit API keys.
- The per-call `results/cache/` and the Python `.venv/` are regenerable and gitignored.
