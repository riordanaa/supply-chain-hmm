# Experiment Notes: HMM Supply Chain Disruption Detection

This document records side-experiments, attempted changes, and running context for the
final sprint of the project. It complements `report.md` (the main academic writeup) and
lives alongside the code so future work has a record of what was tried and why.

---

## 1. Summary of the Project (from `report.md`)

**Thesis.** Give a downstream Distributor (DR) probabilistic situational awareness of an
upstream Manufacturer's (MN) hidden operational state — Steady, Disruption, or Recovery —
using only the lagged, noisy signals the DR can actually observe (its own backlog and
received shipments).

**Network.** Three-echelon pharma chain: MN → DR → HC, with 2-week production lead time,
2-week MN→DR shipping lead time, and 2-week DR→HC shipping lead time. HC demand
$D_t \sim \mathcal{N}(200, 10^2)$ per week.

**HMM.** $N = 3$ hidden states (Steady, Disruption, Recovery), $M = 6$ discrete emissions
= (DR Backlog: 2 levels) × (DR Shipment: 3 levels). Disruption onset is fixed at $t = 15$
(95% capacity reduction). Disruption duration is a Geometric random variable with $p = 0.08$
(mean 12.5 weeks) — deliberately chosen so the recovery transition is memoryless and the
Markov property holds exactly at the hidden-state level.

**Memoryless property analysis.** KS tests on dwell-time distributions and a first-order
vs. second-order Markov likelihood ratio test ($\chi^2 = 175.7$, $df = 12$, $p < 0.001$)
show the natural supply chain violates the memoryless property — motivating the Geometric
encoding.

**Training.** Supervised MLE (direct counting from labeled data, Laplace $\alpha = 1$),
parameters injected into `hmmlearn.CategoricalHMM` for optimized Forward/Viterbi inference.
70 train / 30 test runs of 80 weeks each. A 10-week warmup truncation on training sequences
removes the MN-starts-at-zero startup transient.

**Key results.**
- Learned $\hat{a}_{11} = 0.921 \approx 1 - p$ — the HMM recovers the known Geometric
  parameter from the data, validating both the training procedure and the Geometric encoding.
- Real-time disruption detection lag: **10.2 weeks** mean (≈4 weeks physical propagation
  + ≈6 weeks statistical/algorithmic lag from the strong Steady prior $a_{00} = 0.978$).
- Recovery detection lag: **4.0 weeks** (faster because the shipment surge is distinctive).
- Viterbi overall accuracy: **80.0%** raw; **85.2%** with lead-time-adjusted ground truth;
  disruption precision jumps to **98.4%** after the adjustment.
- Low Disruption recall (23.5%) is driven by physically unobservable micro-disruptions
  (duration ≤ 5 weeks) — a fundamental physical constraint, not a model deficiency.

---

## 2. Summary of the Codebase

Root folder `supply_chain_sim-main_or/` contains a pharmaceutical supply chain simulator
from Raziei's 2024 dissertation. This project does **not** modify the simulator; it wraps
it in the `hmm_project/` subfolder.

### Parent-folder simulator (untouched by this project)

| File | Role |
|------|------|
| `Producer.py` | MN / factory agent. Constructor `__init__(transhippers, name, ss, m, l, pl, ...)` sets `inventory = ss`. Maintains `production_queue` and `allocations_queue` as in-flight pipelines. |
| `Transhipper.py` | DR / Wholesaler agent. Also starts with `inventory = ss`. |
| `Consumer.py` | HC / Health Center agent. |
| `Manufacturer.py`, `Wholesaler.py`, `Hospital.py` | Alternative single-agent wrappers. |
| `Simulation.py` | Orchestrates the per-week step sequence across all agents. |
| `op_base_stock_all_first.py`, `op_proportional.py`, etc. | Pluggable ordering / allocation policies. |
| `pp_maximum_capacity.py`, `pp_base_stock.py` | Production policies. |

### `hmm_project/` — the HMM layer

| File | Role |
|------|------|
| `config.py` | **Central parameter store.** All tunables (N_RUNS, SIM_PERIODS, DISRUPTION_ONSET, GEOM_P, PRODUCTION_MAX, DISRUPTION_FACTOR, safety stock levels, lead times, observation-discretization thresholds, warmup truncation, seeds) live here. |
| `run_simulation.py` | Configures the 1-MN / 1-DR / 1-HC topology, installs the Geometric-duration disruption trigger, runs `N_RUNS` replications, logs per-period signals. |
| `instrumented_agents.py` | Thin subclasses `InstrumentedProducer`, `InstrumentedTransshipper`, etc. — add `h_inventory` / `h_backlog` logging without modifying the parent simulator. |
| `preprocess.py` | Labels ground-truth states (Steady/Disruption/Recovery) from MN production + backlog, discretizes DR observations into the M=6 space, applies `WARMUP_PERIODS` truncation, splits 70/30. |
| `hmm_model.py` | Supervised MLE by direct counting (π, A, B with Laplace smoothing), then injects frozen parameters into `hmmlearn.CategoricalHMM`. |
| `evaluate.py` | Runs Forward & Viterbi on test data; computes detection lag histograms, confusion matrices, per-state precision/recall/F1, lead-time-adjusted metrics, filtered-accuracy metrics. |
| `visualize.py` | Generates all publication figures: `hero_figure.png`, `forward_probabilities.png`, `confusion_matrix.png`, `detection_lag_histogram.png`, `trained_matrices.png`, `raw_signals.png`, `observation_frequency_heatmap.png`. |
| `run_pipeline.py` | Orchestrator: Data Generation → Preprocessing → Train/Eval/Visualize. One-shot end-to-end. |
| `test_memoryless.py` | Stand-alone script that runs 200 fixed-duration-disruption sims and generates `memoryless_property_test.png` (KS + LR tests). |
| `generate_html.py`, `generate_pdf.py` | Render `report.md` to HTML / PDF. |
| `report.md` | The academic writeup. |
| `data/processed/` | Pickled training / test arrays (regenerated each pipeline run). |
| `results/` | Figures and summary JSON (regenerated each pipeline run). |

### Pipeline flow

```
run_pipeline.py
    ├── run_simulation.run_all()    → data/raw arrays of (states, observations) per run
    ├── preprocess.preprocess_all() → label, discretize, warmup-truncate, 70/30 split
    ├── evaluate.run_evaluation()   → train MLE HMM, run Forward/Viterbi, compute metrics
    └── visualize.generate_all_plots() → 7 figures to results/
```

---

## 3. Step 1 Attempt: Initialize MN with Safety Stock (REVERTED)

### Motivation

In the baseline config, `MN_SAFETY_STOCK = 0`: the Manufacturer starts with zero on-hand
inventory and must build up through production while the DR is already placing orders.
This creates a startup transient that structurally resembles a Recovery phase (growing
inventory, shrinking backlog) and necessitates a 10-week `WARMUP_PERIODS` truncation hack
in preprocessing.

**Hypothesis.** Initializing the MN with its target safety stock on hand should eliminate
the startup artifact entirely, letting us remove the warmup truncation and simplify the
report narrative (π would be a clean (1, 0, 0) rather than leaning on truncation).

### What we changed

Two config-only edits, no simulator modifications:

| File | Change |
|------|--------|
| `config.py:20` | `MN_SAFETY_STOCK = 0` → `MN_SAFETY_STOCK = 800` (= 4 weeks × 200 units/wk demand = full pipeline coverage) |
| `config.py:66` | `WARMUP_PERIODS = 10` → `WARMUP_PERIODS = 4` |

### Result: the startup fix worked, but exposed a deeper problem

**Startup transient: eliminated** ✓
- After truncating only 4 weeks, **100 / 100 runs start in Steady** (up from requiring a
  10-week chop in the baseline).
- Learned π = (0.9726, 0.0137, 0.0137) — nearly all probability mass on Steady, as a
  true steady-state start should produce.

**But the HMM collapsed to always predicting Steady.**

Confusion matrix with `MN_SAFETY_STOCK = 800`:

| True \ Pred | Steady | Disruption | Recovery |
|---|---|---|---|
| Steady (1969) | **1969** | 0 | 0 |
| Disruption (357) | **357** | 0 | 0 |
| Recovery (74) | **74** | 0 | 0 |

Overall accuracy = 82.0% = **majority-class baseline**. No disruption detected in any run.
Disruption recall 0 %.

### Why — it's a physics problem, not a modeling bug

With `MN_SAFETY_STOCK = 800`, the MN now has a ~4-week reservoir of inventory. When the
disruption hits at $t = 15$ and production drops from 800 → 40 units/week, the MN
**continues to ship the DR at 200 units/week** from its buffer:

- Drain rate: $200_{\text{demand}} - 40_{\text{production}} = 160$ units/week
- Time to exhaust buffer: $800 / 160 \approx 5$ weeks
- Plus 2-week MN → DR shipping lead time = **7-week total propagation delay**

Under Geom($p = 0.08$), roughly 34 % of disruptions last $\leq 5$ weeks and are
**completely absorbed by the MN buffer** — the DR sees no abnormal signal whatsoever.
Even longer disruptions produce only 1-3 weeks of DR-visible disruption signal late in
the episode. The trained emission matrix reflects this directly: 1197 / 1253 Disruption-
labeled weeks in training emit `obs=1 (None-BL, Normal-Ship)` — the exact same signature
as Steady.

The HMM isn't broken. It correctly learned that Disruption and Steady produce
indistinguishable DR observations 93 % of the time, so the maximum-likelihood Viterbi
path just stays in Steady.

### Why we reverted

1. **It over-complicates the narrative.** The core story of the report — "DR detects
   upstream disruption with 10.2-week lag; supervised MLE recovers $a_{11} \approx 1 - p$"
   — depends on the DR actually being able to see the disruption. With MN_ss = 800 the
   DR can't, and the 6-observation space becomes structurally insufficient.
2. **It would require also doing Step 3.** The only way to recover detection performance
   with a steady-state-initialized MN is to expand the observation space to include an
   upstream leading indicator (the 2×2×2 plan with MN Fill Rate). That coupling makes
   Step 1 non-optional and adds complexity we don't need.
3. **User decision:** simpler is better for the course deliverable. The 10-week warmup
   truncation is a defensible preprocessing step explicable in one sentence; the
   alternative requires restructuring the whole emission space.

### What we reverted to

Both `config.py:20` and `config.py:66` restored to original values (`MN_SAFETY_STOCK = 0`,
`WARMUP_PERIODS = 10`). Re-running the pipeline reproduced every headline number in
`report.md` exactly (10.2-week disruption lag, 80.0 % Viterbi accuracy, 85.2 %
lead-time-adjusted, 98.4 % adjusted disruption precision, $\hat{a}_{11} = 0.921$). No
other files were ever modified.

### Takeaway for the report (if desired)

If useful for future work: *"Initializing the MN with steady-state inventory eliminates
the startup transient but reveals that the 6-observation DR-only view cannot detect
disruptions absorbed by realistic manufacturer buffers. This motivates observation spaces
that include upstream signals."* — a one-sentence bridge to Step 3 that the user may
revisit later.

---

## 4. Step 2: Longer Disruptions ($p = 0.04$, SIM_PERIODS = 150) — COMPLETED

### Config changes

Two single-line edits in `hmm_project/config.py`:

| File:Line | Before | After |
|---|---|---|
| `config.py:8` | `SIM_PERIODS = 80` | `SIM_PERIODS = 150` |
| `config.py:10` | `GEOM_P = 0.08` | `GEOM_P = 0.04` |

Everything else held constant (`DISRUPTION_ONSET = 15`, `WARMUP_PERIODS = 10`,
`MN_SAFETY_STOCK = 0`, observation thresholds, 70/30 split, seeds).

### Archived outputs

To preserve both parameter regimes side-by-side, the outputs were consolidated into
two self-contained subfolders of `hmm_project/`:

- `hmm_project/p08/` — baseline regime
  - `config_snapshot.py` — exact config used (`GEOM_P = 0.08`, `SIM_PERIODS = 80`)
  - `results/` — all 7 EDA figures + `evaluation_results.npz`
  - `data/processed/` — `train.npz`, `test.npz`, `test_metadata.json`
  - `README.md` — short pointer / reproduction instructions
- `hmm_project/p04/` — robustness-check regime
  - `config_snapshot.py` — exact config used (`GEOM_P = 0.04`, `SIM_PERIODS = 150`)
  - `results/` — all 7 EDA figures + `evaluation_results.npz`
  - `data/processed/` — `train.npz`, `test.npz`, `test_metadata.json`
  - `README.md` — short pointer / reproduction instructions

`hmm_project/results/` and `hmm_project/data/processed/` still hold the most recent
pipeline run (currently $p = 0.04$) but the durable archives are the regime folders.

`hmm_project/report.md` was extended with a new Section 7.4 documenting the $p = 0.04$
robustness check; figure paths for the existing p=0.08 content were updated from
`results/...` to `p08/results/...` to match the new folder layout. All other content in
`report.md` is unchanged.

### Headline result: MLE recovers $p = 0.04$ as cleanly as it recovered $p = 0.08$

**Trained transition matrix $\hat{A}$ under $p = 0.04$:**

```
     From / To    Steady  Disruption  Recovery
     Steady      0.9887    0.0099    0.0014
     Disruption  0.0120   [0.9613]   0.0267     ← self-transition ≈ 1 − p = 0.96 ✓
     Recovery    0.0738    0.0013    0.9248
```

Side-by-side comparison of the key diagonal:

| Entry | $p = 0.08$ (baseline, report.md) | $p = 0.04$ (new) | Expected ($= 1 - p$) |
|---|---|---|---|
| $\hat{a}_{11}$ (Disruption self-transition) | 0.9209 | **0.9613** | 0.96 |
| Total exit rate from Disruption ($\hat{a}_{10} + \hat{a}_{12}$) | 0.0791 | **0.0387** | 0.04 |

The MLE nailed the new Geometric parameter to within ~0.001. This is a **second,
independent validation** of the Geometric-encoding approach: the method is not
reproducing a single fitted number by luck — it recovers the correct rate under a
different underlying $p$.

### Data-generation sanity

- Ground-truth state breakdown across all 100 runs (after warmup truncation):
  Steady 72.6%, Disruption 17.1%, Recovery 10.3% — more Disruption mass than before
  (was ~13%) because each disruption episode is twice as long on average.
- Shipment baseline (pre-disruption) = 295.4 (vs. ~184 under the shorter simulation) —
  reflects the longer lead-in period before disruption.
- 100 / 100 runs start in Steady after the 10-period warmup truncation ✓.
- Micro-disruptions (≤ 5 weeks) dropped from 8/30 to **4/30** of test runs —
  consistent with Geom(0.04) putting less probability mass on short durations.

### Viterbi classification quality improved substantially

| Metric | $p = 0.08$ (report.md) | $p = 0.04$ |
|---|---|---|
| Overall accuracy | 80.0% | **86.8%** |
| Majority-class baseline | 72.2% | 73.9% |
| Improvement over baseline | +7.8 pp | **+12.9 pp** |
| Lead-time-adjusted accuracy | 85.2% | **93.7%** |
| Lead-time-adjusted Disruption precision | 98.4% | **100.0%** |
| Lead-time-adjusted Disruption recall | 33.9% | **76.6%** |
| Filtered accuracy (excluding micro-disruptions) | 76.0% | **85.7%** |
| Disruption-detection runs (Forward $P > 0.5$) | 20 / 30 | **24 / 30** |

The jump in lead-time-adjusted Disruption recall (33.9 % → 76.6 %) is the most
striking change: longer disruptions spend more weeks in the DR-observable window
once the physical signal propagates, so the Viterbi path spends proportionally more
time correctly labeling them as Disruption rather than being "swallowed" at the edges.

### Physical detection lag is unchanged (as expected)

- Disruption detection lag: **10.2 weeks** in both regimes.
- Recovery detection lag: **4.0 weeks** in both regimes.

Both lags are dictated by lead-time propagation (production 2 wk + shipping 2 wk ≈ 4 wk
physical + ~6 wk statistical before $P(\text{Disruption}) > 0.5$), which is a function
of the physical pipeline — not of $p$. The fact that the lag didn't move under a
doubled disruption duration is itself a corroboration of the physical-vs-statistical
decomposition in Section 7.1 of `report.md`.

### Emission matrix also sharper

$\hat{B}$ (rows = state, cols = obs 0..5):

```
            obs=0   obs=1   obs=2   obs=3   obs=4   obs=5
Steady     0.2230  0.7472  0.0165  0.0001  0.0008  0.0124
Disruption 0.0528  0.2705  0.0005  0.6739  0.0016  0.0005
Recovery   0.0013  0.0160  0.0013  0.2503  0.0053  0.7257
```

Key differences from the $p = 0.08$ emission matrix:

- Disruption → obs=3 (*High-BL, Zero/Low-Ship*): **0.674** (was 0.445). Much sharper —
  longer disruptions spend more periods in the fully-propagated "DR sees the shock"
  regime.
- Recovery → obs=5 (*High-BL, Surge-Ship*): **0.726** (was 0.514). Longer recoveries
  spend more time visibly clearing the accumulated backlog.
- Steady emits obs=0 more frequently (0.223 vs. 0.019) — a side effect of the longer
  150-week run producing more periods where inventory is momentarily low before a
  production tick lands. Does not meaningfully affect discriminability.

### Plots regenerated

All seven EDA figures were regenerated automatically by `visualize.py` using the new
$p = 0.04$ data and are archived in `hmm_project/p04/results/`:

- `hero_figure.png` — Forward filtering on a test run with the longer sim window
- `raw_signals.png` — single-run time series (x-axis now spans 150 weeks)
- `observation_frequency_heatmap.png` — emission heatmap with sharper diagonal
- `trained_matrices.png` — $\hat{A}$ and $\hat{B}$ visualizations showing the 0.96 self-loop
- `confusion_matrix.png` — Viterbi confusion (much stronger diagonal)
- `detection_lag_histogram.png` — detection lag distributions (24/30 runs detect now)
- `forward_probabilities.png` — Forward filter probabilities across test set

The `memoryless_property_test.png` is unchanged (the memoryless analysis is
independent of `GEOM_P` — it probes the natural non-Geometric system) and is present
in both archive folders as a reference.

### Bottom line

Both key predictions of the plan hold:

1. The MLE recovers the new Geometric parameter ($\hat{a}_{11} = 0.9613 \approx 0.96$,
   total exit rate $= 0.0387 \approx 0.04$).
2. Longer disruptions improve detection quality across the board without changing
   the physical lag.

The $p = 0.04$ regime gives a **cleaner, more compelling story** than the baseline:
93.7% lead-time-adjusted accuracy and 100% Disruption precision vs. 85.2% / 98.4%.
These results now live in `report.md` Section 7.4 (*Robustness Check: Halved
Disruption Probability*) alongside the original $p = 0.08$ writeup.

---

## 5. Step 3: Recovery vs. Disruption Duration Scatter (Deterministic Clearing Rate) — COMPLETED

### Goal

Build a scatter plot of Recovery Duration vs. Disruption Duration for all 100 runs
with a linear trendline and a theoretical reference slope, then embed it in a new
subsection of the report (Section 4.4: *Deterministic Recovery Clearing Rate*).

### Code additions

- New functions in `visualize.py`:
  - `plot_recovery_vs_disruption(processed_dir, output_path, ...)` — single-regime
    scatter with empirical fit (OLS) and theoretical slope reference.
  - `plot_recovery_vs_disruption_combined(regime_configs, output_path, ...)` —
    overlays multiple regimes on one set of axes.
  - Helpers `_load_all_state_sequences`, `_durations_per_run`, `_fit_line`,
    `_is_truncated` for reuse.
  - Call added to `generate_all_plots()` so future pipeline runs produce
    `results/recovery_vs_disruption.png` by default.
- New standalone module `regression_diagnostics.py` that runs the full OLS
  assumption battery (Breusch–Pagan, Shapiro–Wilk, Durbin–Watson, Ramsey RESET)
  on each regime's filtered data and saves a 4-panel diagnostic figure per regime
  (residuals-vs-fitted, Normal Q-Q, Scale-Location, Residuals-vs-Leverage with
  Cook's distance contours). Summary dumped to
  `results/regression_diagnostics_summary.json`.

### Outputs

- `hmm_project/p08/results/recovery_vs_disruption.png`
- `hmm_project/p04/results/recovery_vs_disruption.png`
- `hmm_project/results/recovery_vs_disruption_combined.png`
- `hmm_project/p08/results/regression_diagnostics.png`
- `hmm_project/p04/results/regression_diagnostics.png`
- `hmm_project/results/regression_diagnostics_summary.json`

### Key findings

**A small number of runs (3 under $p=0.08$, 5 under $p=0.04$) had disruptions
long enough to be right-censored** by `SIM_PERIODS` — the simulation ended while
the factory was still in Disruption or Recovery, giving an observed Recovery
Duration that is a strict lower bound on the true value (typically 0). Criterion
for flagging: final state ≠ Steady. These runs are excluded from the fit
because they systematically bias the slope downward.

**After filtering, the linear relationship is clearly real:**

| Metric | $p = 0.08$ | $p = 0.04$ |
|---|---|---|
| n (after filter) | 97 / 100 | 95 / 100 |
| Slope | 0.582 | 0.535 |
| Intercept | −0.11 | +0.44 |
| $R^2$ | 0.811 | 0.939 |
| Breusch–Pagan (homoscedasticity) | $p = 1.00$ ✓ | $p = 0.48$ ✓ |
| Ramsey RESET (linearity) | $p = 0.36$ ✓ | $p = 0.033$ ⚠️ (marginal) |
| Cook's D (max) | 0.076 ✓ | 0.137 ✓ |
| Shapiro–Wilk (residual normality) | $p < 10^{-3}$ ⚠️ | $p < 10^{-3}$ ⚠️ |

Normality rejection is driven by a floor cluster at small $D_{\text{disr}}$ where
the MN backlog barely crosses the 50-unit Recovery-state threshold and pins
$D_{\text{rec}}$ at ~0 or 5. This affects CI inference on the slope but not the
OLS point estimate itself.

**The empirical slope (~0.55) differs from my naive MN-only theoretical
prediction (0.267).** The naive calculation — $(200 - 40)/(800 - 200) = 160/600$
— treats the MN in isolation. In reality the clearing rate is set by the
coupled MN/DR system: once the DR's 500-unit safety stock is depleted within
~3 weeks of disruption onset, its base-stock policy orders up to 500 units/wk
from the MN, not just 200. This amplifies backlog accumulation beyond 160/wk
during disruption and keeps DR demand elevated during recovery (so the MN
clears at less than 600/wk). An extreme coupled-system upper bound assuming
sustained 500-unit/wk DR ordering throughout would give
$(500 - 40) / (800 - 500) \approx 1.53$; the observed 0.55 lies in between.

**Crucially, the two regimes agree.** Slopes of 0.582 ($p=0.08$) and 0.535
($p=0.04$) differ by under 10%, consistent with sampling noise on ~95 runs.
This is the key finding: the clearing rate is a function of physical
parameters (capacities, safety stocks, lead times) and does **not** depend on
$p$. This is the same pattern as the $\hat{a}_{11}$ cross-regime recovery —
structural validation at the level of physical dynamics.

The scatter plots and these findings now live in `report.md` Section 4.4
(*Deterministic Recovery Clearing Rate*). Figure numbering in the existing
Section 7.4 was bumped from 7–9 to 10–12 to accommodate the new Figures 7–9.

### Presentation advice note

During review the user asked whether presenting the naive-then-coupled arc
(start with slope $= 0.267$, observe empirical $\approx 0.55$, then explain
via DR base-stock reordering dynamics) was overcomplicating things versus
just deriving the $\approx 0.5$ slope up front. Written answer captured for
reference:

- **Keep the arc in the written report** — the journey is pedagogically
  defensible and the coupled-system derivation is not a one-line result.
- **For the slide deck, lead with regime-invariance** — the headline is
  "two regimes produce the same slope (0.58 vs. 0.54)," not "the slope
  value is X." Whether the slope equals 0.267 or 0.55 is secondary; the
  fact that **it is** $p$-invariant is what the project demonstrates.
- **Coupled explanation belongs as a footnote / backup slide**, expanded
  only if a sharp audience member asks why empirical differs from naive.

---

## 6. Cosmetic Cleanup: Viterbi Confusion Matrix Restyling — COMPLETED

Pure cosmetic change, no data or model touched:

- `plot_confusion_matrix()` in `visualize.py`: `cmap='YlOrRd'` → `cmap='Blues'`
  (matches the 3×3 Transition Matrix $A$ styling in `trained_matrices.png`).
- `figsize=(7, 6)` → `figsize=(5.5, 4.5)` (~30% smaller); font sizes trimmed
  slightly for the smaller frame.

All three archived PNGs regenerated from the saved
`evaluation_results.npz` arrays (no re-evaluation). Verified every cell
value matches `report.md` Sections 6.4 and 7.4 to the integer:

- $p = 0.08$ baseline: diag 1630 / 84 / 206, row totals 1734 / 357 / 309,
  overall accuracy 80.0%.
- $p = 0.04$ robustness: diag 3216 / 464 / 227, row totals 3326 / 731 / 443,
  overall accuracy 86.8%.

Output files overwritten in place (so `report.md` figure paths keep resolving
to the restyled PNGs):
- `hmm_project/p08/results/confusion_matrix.png`
- `hmm_project/p04/results/confusion_matrix.png`
- `hmm_project/results/confusion_matrix.png`

---

## 7. Final Project Status

The course project is essentially feature-complete. Below is a single place
to see what's done versus what's intentionally deferred.

### Completed

| Step | Status | Home in `report.md` | Archive |
|---|---|---|---|
| Step 0 — baseline HMM pipeline (MN→DR→HC, $N=3$, $M=6$) | ✓ | Sections 1–7.3 | `p08/` |
| Step 1 — MN safety-stock initialization | **reverted** | not in report | — |
| Step 2 — halved $p$ robustness check ($p=0.04$, `SIM_PERIODS=150`) | ✓ | Section 7.4 | `p04/` |
| Step 3 — recovery-vs-disruption scatter (deterministic clearing rate) | ✓ | Section 4.4 | `{p08,p04}/results/recovery_vs_disruption.png`, `results/recovery_vs_disruption_combined.png` |
| Regression-assumption battery on the scatter | ✓ | summarized in Section 4.4; panels archived | `{p08,p04}/results/regression_diagnostics.png`, `results/regression_diagnostics_summary.json` |
| Confusion-matrix restyling (Blues, compact) | ✓ | Figures 6 and 12 | archived |
| Code committed + pushed to GitHub | ✓ | `https://github.com/riordanaa/supply-chain-hmm` | master branch |

Final commit chain on `master`:
- `b6f7d99` — original baseline snapshot (pre-project work)
- `f64fb8c` — Step 2: $p=0.04$ robustness + `p08/` & `p04/` consolidation
- `6092df6` — Step 3: scatter + deterministic clearing rate Section 4.4
- `5be1e09` — Viterbi confusion matrix Blues restyle

### Intentionally out of scope (documented so future-you doesn't
                                   wonder why they're missing)

| Not done | Reason |
|---|---|
| 2×2×2 = $M=8$ observation space with MN Fill Rate as a leading indicator | Out of scope per the user's Step 3 triage — would be valuable for beating the 10.2-week detection lag but is its own project. |
| Renewal-theory / expected-remaining-duration treatment | Substituted with the existing Geometric self-transition argument ($\text{E}[\text{remaining}] = 1/(1 - \hat{a}_{11})$); see `report.md` Section 2.5. |
| Optimal threshold policy / POMDP / SARSA on HMM output | Out of scope; mentioned only qualitatively in the Conclusion / Discussion. |
| Second distributor (DR2) | Out of scope — would break the current 1-MN/1-DR/1-HC wrapper and require changes to the underlying simulator. |

### Useful commands for future-you

Reproduce either regime from scratch:
```bash
cp hmm_project/p08/config_snapshot.py hmm_project/config.py   # or p04/
python hmm_project/run_pipeline.py
```

Regenerate only the scatter + combined:
```bash
cd hmm_project
python -c "from visualize import plot_recovery_vs_disruption, plot_recovery_vs_disruption_combined; \
  plot_recovery_vs_disruption('p08/data/processed', 'p08/results/recovery_vs_disruption.png', title_suffix=' (p = 0.08)'); \
  plot_recovery_vs_disruption('p04/data/processed', 'p04/results/recovery_vs_disruption.png', title_suffix=' (p = 0.04)'); \
  plot_recovery_vs_disruption_combined([ \
    {'processed_dir':'p08/data/processed','label':'p = 0.08','color':'#3498db'}, \
    {'processed_dir':'p04/data/processed','label':'p = 0.04','color':'#e67e22'}], \
    'results/recovery_vs_disruption_combined.png')"
```

Rerun the regression-assumption diagnostics:
```bash
cd hmm_project && python regression_diagnostics.py
```

Convert the report to PDF / HTML for submission:
```bash
cd hmm_project && python generate_pdf.py   # or generate_html.py
```
