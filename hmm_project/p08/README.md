# $p = 0.08$ — Baseline Run

This folder archives the inputs, outputs, and figures for the **baseline** parameter
regime documented in Sections 1-7 of `../report.md`.

- $p = 0.08$ — Geometric disruption recovery probability per week
- Expected disruption duration: $1/p = 12.5$ weeks
- `SIM_PERIODS = 80` weeks per run

## Contents

| Path | What |
|------|------|
| `config_snapshot.py` | Exact config used for this run. Copy to `../config.py` to reproduce. |
| `results/` | All 7 EDA figures + `evaluation_results.npz` summary metrics. |
| `data/processed/` | Pickled `train.npz`, `test.npz`, and `test_metadata.json` from the 70/30 split. |

## Reproduce

```bash
cp hmm_project/p08/config_snapshot.py hmm_project/config.py
python hmm_project/run_pipeline.py
```

## Headline results (see `../report.md` for full discussion)

- Trained Disruption self-transition $\hat{a}_{11} = 0.921 \approx 1 - p$
- Disruption detection lag: 10.2 weeks mean
- Viterbi overall accuracy: 80.0% (+7.8 pp over baseline)
- Lead-time-adjusted accuracy: 85.2%; adjusted Disruption precision: 98.4%
