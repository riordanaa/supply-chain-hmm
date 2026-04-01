"""
Full pipeline: generate data -> preprocess -> train -> evaluate -> visualize.
Run this script to execute everything end-to-end.
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

print("=" * 70)
print("HMM SUPPLY CHAIN DISRUPTION DETECTION — FULL PIPELINE")
print("=" * 70)

# --- Phase 1: Data Generation ---
print("\n" + "=" * 70)
print("PHASE 1: DATA GENERATION (100 simulation runs)")
print("=" * 70)
t0 = time.time()
from run_simulation import run_all
run_all(verbose=True)
print(f"Data generation completed in {time.time() - t0:.1f} seconds.")

# --- Phase 2: Preprocessing ---
print("\n" + "=" * 70)
print("PHASE 2: PREPROCESSING (ground-truth labeling, discretization, split)")
print("=" * 70)
t0 = time.time()
from preprocess import preprocess_all
preprocess_all(verbose=True)
print(f"Preprocessing completed in {time.time() - t0:.1f} seconds.")

# --- Phase 3-5: Train, Evaluate, Visualize ---
print("\n" + "=" * 70)
print("PHASES 3-5: TRAIN (supervised MLE) -> EVALUATE -> VISUALIZE")
print("=" * 70)
t0 = time.time()
from evaluate import run_evaluation
from visualize import generate_all_plots

results = run_evaluation(verbose=True)
generate_all_plots(results)
print(f"Training + evaluation + visualization completed in {time.time() - t0:.1f} seconds.")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"Results and plots saved to: hmm_project/results/")
