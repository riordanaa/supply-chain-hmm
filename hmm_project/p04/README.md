# $p = 0.04$ — Robustness Check (Halved Disruption Probability)

This folder archives the inputs, outputs, and figures for the **robustness-check**
parameter regime documented in Section 7.4 of `../report.md`.

- $p = 0.04$ — Geometric disruption recovery probability per week (half of baseline)
- Expected disruption duration: $1/p = 25$ weeks
- `SIM_PERIODS = 150` weeks per run (extended from 80 to ensure disruptions + recovery
  complete within the simulation window)
- All other parameters identical to the $p = 0.08$ baseline.

## Why this run exists

The baseline run showed $\hat{a}_{11} = 0.921 \approx 1 - 0.08$ — the supervised MLE
recovers the known Geometric parameter from training data. Re-running under a different
$p$ tests whether this is a genuine recovery of the rate, or an artifact specific to
one fitted number. The result: $\hat{a}_{11} = 0.9613 \approx 1 - 0.04$, confirming that
the method generalizes.

## Contents

| Path | What |
|------|------|
| `config_snapshot.py` | Exact config used for this run. Copy to `../config.py` to reproduce. |
| `results/` | All 7 EDA figures + `evaluation_results.npz` summary metrics. |
| `data/processed/` | Pickled `train.npz`, `test.npz`, and `test_metadata.json` from the 70/30 split. |

## Reproduce

```bash
cp hmm_project/p04/config_snapshot.py hmm_project/config.py
python hmm_project/run_pipeline.py
```

## Headline results (see `../report.md` Section 7.4 for full discussion)

- Trained Disruption self-transition $\hat{a}_{11} = 0.9613 \approx 1 - p$
- Total exit rate from Disruption $= 0.0387 \approx p$
- Disruption detection lag: 10.2 weeks (unchanged — physical constraint)
- Viterbi overall accuracy: 86.8% (+12.9 pp over baseline)
- Lead-time-adjusted accuracy: 93.7%; adjusted Disruption precision: 100.0%,
  adjusted Disruption recall: 76.6% (up from 33.9%)
