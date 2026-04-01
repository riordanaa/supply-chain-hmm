"""
Preprocessing pipeline:
1. Load raw CSV data from simulation runs
2. Assign ground-truth hidden states (Steady, Disruption, Recovery)
3. Discretize DR observations into M=6 categories
4. Split into training (70%) and testing (30%) sets
5. Save processed data as .npz files
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter

from config import (
    N_RUNS, SIM_PERIODS, PRODUCTION_MAX, DISRUPTION_ONSET,
    STEADY, DISRUPTION, RECOVERY, STATE_NAMES,
    RECOVERY_BACKLOG_THRESHOLD,
    BACKLOG_HIGH_THRESHOLD, SHIPMENT_LOW_FRACTION, SHIPMENT_SURGE_FRACTION,
    N_OBS, OBS_NAMES, TRAIN_FRACTION, WARMUP_PERIODS
)


def load_raw_data(n_runs=None):
    """Load all raw CSV files and metadata."""
    if n_runs is None:
        n_runs = N_RUNS

    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    runs = []
    metadata_list = []

    for run_id in range(n_runs):
        csv_path = os.path.join(raw_dir, f"run_{run_id:03d}.csv")
        meta_path = os.path.join(raw_dir, f"run_{run_id:03d}_meta.json")

        df = pd.read_csv(csv_path)
        with open(meta_path) as f:
            meta = json.load(f)

        runs.append(df)
        metadata_list.append(meta)

    return runs, metadata_list


def assign_ground_truth(df, original_max):
    """
    Assign ground-truth hidden states based on MN status.

    STEADY (0):     MN at full capacity AND backlog low
    DISRUPTION (1): MN capacity reduced (production_max < original)
    RECOVERY (2):   MN capacity restored BUT backlog still high
    """
    states = np.zeros(len(df), dtype=int)

    for t in range(len(df)):
        prod_max = df["mn_production_max"].iloc[t]
        mn_backlog = df["mn_backlog"].iloc[t]

        if prod_max < original_max:
            states[t] = DISRUPTION
        elif mn_backlog > RECOVERY_BACKLOG_THRESHOLD:
            states[t] = RECOVERY
        else:
            states[t] = STEADY

    return states


def discretize_observations(df, normal_shipment_baseline):
    """
    Discretize DR observations into M=6 categories.

    Backlog (2 levels): None (0), High (1)
    Shipment (3 levels): Zero/Low (0), Normal (1), Surge (2)
    Combined: obs = backlog_level * 3 + shipment_level
    """
    obs = np.zeros(len(df), dtype=int)

    for t in range(len(df)):
        dr_backlog = df["dr_backlog"].iloc[t]
        dr_shipment = df["dr_shipment_received"].iloc[t]

        # Backlog level
        if dr_backlog > BACKLOG_HIGH_THRESHOLD:
            bl_level = 1  # High
        else:
            bl_level = 0  # None

        # Shipment level (relative to baseline)
        if normal_shipment_baseline > 0:
            ship_ratio = dr_shipment / normal_shipment_baseline
        else:
            ship_ratio = 0

        if ship_ratio < SHIPMENT_LOW_FRACTION:
            ship_level = 0  # Zero/Low
        elif ship_ratio > SHIPMENT_SURGE_FRACTION:
            ship_level = 2  # Surge
        else:
            ship_level = 1  # Normal

        obs[t] = bl_level * 3 + ship_level

    return obs


def compute_shipment_baseline(df, onset):
    """
    Compute normal shipment baseline from pre-disruption steady state.
    Skip the first few periods (warmup) where shipments may be 0.
    """
    warmup = 5  # skip first 5 periods for system to stabilize
    pre_disruption = df["dr_shipment_received"].iloc[warmup:onset]
    if len(pre_disruption) > 0:
        return pre_disruption.mean()
    else:
        return 200.0  # fallback to demand mean


def preprocess_all(n_runs=None, verbose=True):
    """Full preprocessing pipeline."""
    runs, metadata_list = load_raw_data(n_runs)
    n = len(runs)

    if verbose:
        print(f"Loaded {n} simulation runs.")

    all_states = []
    all_obs = []
    all_baselines = []

    for i, (df, meta) in enumerate(zip(runs, metadata_list)):
        onset = meta["onset"]

        # Compute shipment baseline
        baseline = compute_shipment_baseline(df, onset)
        all_baselines.append(baseline)

        # Assign ground truth
        states = assign_ground_truth(df, PRODUCTION_MAX)
        all_states.append(states)

        # Discretize observations
        obs = discretize_observations(df, baseline)
        all_obs.append(obs)

    # --- Print diagnostics ---
    if verbose:
        # State distribution
        all_states_flat = np.concatenate(all_states)
        total = len(all_states_flat)
        print(f"\nGround-Truth State Distribution:")
        for s in [STEADY, DISRUPTION, RECOVERY]:
            count = np.sum(all_states_flat == s)
            print(f"  {STATE_NAMES[s]:12s}: {count:6d} periods ({100 * count / total:.1f}%)")

        # Observation distribution
        all_obs_flat = np.concatenate(all_obs)
        print(f"\nObservation Distribution:")
        for o in range(N_OBS):
            count = np.sum(all_obs_flat == o)
            print(f"  {OBS_NAMES[o]:25s} (obs={o}): {count:5d} ({100 * count / total:.1f}%)")

        # Cross-tabulation: state vs observation
        print(f"\nState x Observation Cross-Tabulation (counts):")
        print(f"{'':>12s}", end="")
        for o in range(N_OBS):
            print(f"  obs={o:d}", end="")
        print()
        for s in [STEADY, DISRUPTION, RECOVERY]:
            print(f"{STATE_NAMES[s]:>12s}", end="")
            mask_s = all_states_flat == s
            for o in range(N_OBS):
                mask_o = all_obs_flat == o
                count = np.sum(mask_s & mask_o)
                print(f"  {count:5d}", end="")
            print()

        # Shipment baselines
        print(f"\nShipment Baselines: mean={np.mean(all_baselines):.1f}, "
              f"std={np.std(all_baselines):.1f}")

    # --- Truncate warmup from training sequences ---
    # Drop first WARMUP_PERIODS (10) to remove the artificial backlog artifact
    # caused by MN starting with 0 safety stock. Keeps 4 clean steady-state
    # weeks (t=10-14) before disruption onset (t=15), so π ≈ 100% Steady.
    all_states_truncated = [s[WARMUP_PERIODS:] for s in all_states]
    all_obs_truncated = [o[WARMUP_PERIODS:] for o in all_obs]

    if verbose:
        # Show truncated initial state distribution
        first_states = [s[0] for s in all_states_truncated]
        print(f"\nAfter warmup truncation ({WARMUP_PERIODS} periods dropped):")
        for st in [STEADY, DISRUPTION, RECOVERY]:
            count = sum(1 for s in first_states if s == st)
            print(f"  First state = {STATE_NAMES[st]:12s}: {count}/{n} runs ({100*count/n:.0f}%)")

    # --- Train/Test Split ---
    n_train = int(n * TRAIN_FRACTION)
    train_states = all_states_truncated[:n_train]  # truncated for clean MLE
    train_obs = all_obs_truncated[:n_train]
    test_states = all_states[n_train:]             # FULL sequences for evaluation
    test_obs = all_obs[n_train:]
    test_metadata = metadata_list[n_train:]

    if verbose:
        print(f"\nTrain: {n_train} runs (truncated to {SIM_PERIODS - WARMUP_PERIODS} periods), "
              f"Test: {n - n_train} runs (full {SIM_PERIODS} periods)")

    # --- Save ---
    processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    np.savez(os.path.join(processed_dir, "train.npz"),
             states=np.array(train_states, dtype=object),
             observations=np.array(train_obs, dtype=object))

    np.savez(os.path.join(processed_dir, "test.npz"),
             states=np.array(test_states, dtype=object),
             observations=np.array(test_obs, dtype=object))

    # Save test metadata for evaluation
    import json as json_mod
    with open(os.path.join(processed_dir, "test_metadata.json"), "w") as f:
        json_mod.dump(test_metadata, f, indent=2)

    if verbose:
        print(f"Saved processed data to: {processed_dir}")

    return train_states, train_obs, test_states, test_obs, test_metadata


if __name__ == "__main__":
    preprocess_all()
