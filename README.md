# Probabilistic Phase Recognition in Pharmaceutical Supply Chains via HMM

A course project for Probabilistic and Stochastic Processes. We build a Hidden Markov Model to detect the operational phase (Steady State, Disruption, Recovery) of an upstream Manufacturer from noisy signals observed by a downstream Distributor.

## Project Structure

### Original Simulator (unchanged)

The base supply chain simulator is from Noah Chicoine, based on Dr. Zohreh Raziei's 2024 dissertation. **No original files were modified.**

| File | Description |
|------|-------------|
| `Simulation.py` | Core simulation loop |
| `Producer.py` | Manufacturer agent |
| `Transhipper.py` | Distributor/Wholesaler agent |
| `Consumer.py` | Health Center agent |
| `main.py` | Original entry point with example network configuration |
| `app.py` | Dash web interface |
| `pp_*.py` | Production policies (base stock, max capacity) |
| `ap_proportional.py` | Proportional allocation policy |
| `op_*.py` | Ordering policies (base stock, constant, fulfillment-rate adjusted) |

### HMM Project (new files)

All new code lives in `hmm_project/`. It wraps the existing simulator without modifying it.

| File | Description |
|------|-------------|
| `config.py` | All tunable constants (network params, thresholds, seeds) |
| `instrumented_agents.py` | Subclasses of `Transhipper` and `Producer` that fix copy-semantics bugs in history tracking and add missing backlog recording |
| `run_simulation.py` | Data generation wrapper: configures a 1-MN/1-DR/1-HC network with geometric disruption duration, runs 100 replications |
| `preprocess.py` | Assigns ground-truth hidden states, discretizes DR observations into 6 categories, splits into 70/30 train/test |
| `hmm_model.py` | Supervised MLE training (direct counting) + `hmmlearn.CategoricalHMM` for Forward/Viterbi inference |
| `evaluate.py` | Runs inference on test data, computes detection lag, Viterbi accuracy, confusion matrix |
| `visualize.py` | Generates all publication-quality plots |
| `test_memoryless.py` | Tests whether the supply chain naturally satisfies the Markov memoryless property |
| `run_pipeline.py` | End-to-end orchestrator (generate data -> preprocess -> train -> evaluate -> visualize) |
| `report.md` | Full project report with embedded figures and mathematical derivations |

### Results

Pre-generated results are in `hmm_project/results/`:

| File | Description |
|------|-------------|
| `hero_figure.png` | Ground truth vs Viterbi vs Forward probabilities for a single test run |
| `trained_matrices.png` | Heatmaps of the learned transition (A) and emission (B) matrices |
| `confusion_matrix.png` | 3x3 Viterbi classification confusion matrix |
| `detection_lag_histogram.png` | Distribution of disruption/recovery detection lag |
| `forward_probabilities.png` | Forward-filtered state probabilities over time |
| `raw_signals.png` | Raw DR shipment and backlog signals with state-colored background |
| `memoryless_property_test.png` | Dwell time analysis and Markov order test |

### Data

Simulation output is in `hmm_project/data/`:
- `raw/` — 100 CSV files (one per simulation run) + metadata JSONs
- `processed/` — Train/test `.npz` files with discretized observations and ground-truth labels

## Quick Start

```bash
pip install numpy pandas matplotlib scipy hmmlearn

cd hmm_project
python run_pipeline.py   # runs everything end-to-end (~5 seconds)
```

## Key Results

| Metric | Value |
|--------|-------|
| Viterbi classification accuracy | 80.0% |
| Disruption detection lag (Forward, P>0.5) | 10.2 weeks |
| Recovery detection lag | 4.0 weeks |
| A[Disruption -> Disruption] | 0.921 (matches 1-p = 0.92) |

## Citation

Original simulator by Noah Chicoine (https://github.com/ncc1203). If using this simulator for research, please cite it using the repo link and his name.
