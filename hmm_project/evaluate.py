"""
Evaluation pipeline:
1. Load trained HMM and test data
2. Run Forward algorithm for real-time inference on each test run
3. Run Viterbi algorithm for historical classification
4. Compute metrics: detection lag, Viterbi accuracy, confusion matrix
"""

import os
import json
import numpy as np
from collections import defaultdict

from config import (
    STEADY, DISRUPTION, RECOVERY, STATE_NAMES,
    N_OBS, DISRUPTION_ONSET
)
from hmm_model import SupervisedHMM


def load_processed_data():
    """Load train and test data from processed directory."""
    processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")

    train_data = np.load(os.path.join(processed_dir, "train.npz"), allow_pickle=True)
    test_data = np.load(os.path.join(processed_dir, "test.npz"), allow_pickle=True)

    with open(os.path.join(processed_dir, "test_metadata.json")) as f:
        test_metadata = json.load(f)

    return (
        list(train_data["states"]), list(train_data["observations"]),
        list(test_data["states"]), list(test_data["observations"]),
        test_metadata
    )


def compute_detection_lag(filtered_probs, true_states, target_state, threshold=0.5):
    """
    Compute how many periods after a state first appears in ground truth
    until the Forward algorithm pushes P(state) above threshold.

    Returns lag (int) or None if threshold never exceeded.
    """
    # Find first period where target state appears in ground truth
    first_true = None
    for t in range(len(true_states)):
        if true_states[t] == target_state:
            first_true = t
            break

    if first_true is None:
        return None  # State never appears

    # Find first period at or after first_true where P(state) > threshold
    for t in range(first_true, len(filtered_probs)):
        if filtered_probs[t, target_state] > threshold:
            return t - first_true

    return None  # Threshold never exceeded


def compute_viterbi_metrics(viterbi_paths, true_state_sequences):
    """
    Compute Viterbi classification metrics.

    Returns:
        accuracy: overall fraction of correctly classified periods
        confusion_matrix: (3x3) array
        per_state_metrics: dict with precision, recall, F1 per state
    """
    all_true = np.concatenate(true_state_sequences)
    all_pred = np.concatenate(viterbi_paths)

    # Overall accuracy
    accuracy = np.mean(all_true == all_pred)

    # Confusion matrix
    n_states = 3
    cm = np.zeros((n_states, n_states), dtype=int)
    for true, pred in zip(all_true, all_pred):
        cm[true, pred] += 1

    # Per-state precision, recall, F1
    per_state = {}
    for s in range(n_states):
        tp = cm[s, s]
        fp = cm[:, s].sum() - tp
        fn = cm[s, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_state[STATE_NAMES[s]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": cm[s, :].sum()
        }

    return accuracy, cm, per_state


def run_evaluation(verbose=True):
    """Full evaluation pipeline."""
    # Load data
    train_states, train_obs, test_states, test_obs, test_metadata = load_processed_data()

    if verbose:
        print(f"Training on {len(train_states)} runs, testing on {len(test_states)} runs.")

    # --- Train model ---
    model = SupervisedHMM()
    model.train(train_states, train_obs)

    if verbose:
        model.print_parameters()

    # --- Inference on test set ---
    all_filtered = []      # Forward algorithm results
    all_viterbi = []       # Viterbi paths
    disruption_lags = []   # Detection lag for disruption
    recovery_lags = []     # Detection lag for recovery

    for i, (true_s, obs_s, meta) in enumerate(zip(test_states, test_obs, test_metadata)):
        # Forward algorithm (real-time filtered probabilities)
        filtered = model.forward_probabilities(obs_s)
        all_filtered.append(filtered)

        # Viterbi algorithm (most likely path)
        viterbi_path, log_prob = model.viterbi(obs_s)
        all_viterbi.append(viterbi_path)

        # Detection lag for disruption
        lag_d = compute_detection_lag(filtered, true_s, DISRUPTION, threshold=0.5)
        if lag_d is not None:
            disruption_lags.append(lag_d)

        # Detection lag for recovery
        lag_r = compute_detection_lag(filtered, true_s, RECOVERY, threshold=0.5)
        if lag_r is not None:
            recovery_lags.append(lag_r)

    # --- Viterbi metrics ---
    accuracy, cm, per_state = compute_viterbi_metrics(all_viterbi, test_states)

    # --- Print results ---
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Detection lag
        print("\n--- Real-Time Detection Lag (Forward Algorithm) ---")
        if disruption_lags:
            print(f"  Disruption detection (P>0.5):")
            print(f"    Mean lag: {np.mean(disruption_lags):.1f} weeks")
            print(f"    Median:   {np.median(disruption_lags):.1f} weeks")
            print(f"    Min: {min(disruption_lags)}, Max: {max(disruption_lags)}")
            print(f"    Runs with detection: {len(disruption_lags)}/{len(test_states)}")
        else:
            print("  No disruption detected in any test run!")

        if recovery_lags:
            print(f"  Recovery detection (P>0.5):")
            print(f"    Mean lag: {np.mean(recovery_lags):.1f} weeks")
            print(f"    Median:   {np.median(recovery_lags):.1f} weeks")
            print(f"    Min: {min(recovery_lags)}, Max: {max(recovery_lags)}")
            print(f"    Runs with detection: {len(recovery_lags)}/{len(test_states)}")
        else:
            print("  No recovery detected in any test run!")

        # Also compute lags at higher thresholds
        for thresh in [0.7, 0.9]:
            lags_t = []
            for filtered, true_s in zip(all_filtered, test_states):
                lag = compute_detection_lag(filtered, true_s, DISRUPTION, threshold=thresh)
                if lag is not None:
                    lags_t.append(lag)
            if lags_t:
                print(f"  Disruption detection (P>{thresh}): mean={np.mean(lags_t):.1f} weeks "
                      f"({len(lags_t)}/{len(test_states)} runs)")

        # Viterbi accuracy
        print(f"\n--- Viterbi Classification Accuracy ---")
        print(f"  Overall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

        # Majority class baseline
        all_true = np.concatenate(test_states)
        majority = max(Counter(all_true).values()) / len(all_true)
        print(f"  Majority class baseline: {majority:.4f} ({majority*100:.1f}%)")
        print(f"  Improvement over baseline: {(accuracy - majority)*100:.1f} percentage points")

        # Confusion matrix
        print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
        print(f"{'':>14s}", end="")
        for j in range(3):
            print(f"  {STATE_NAMES[j]:>12s}", end="")
        print()
        for i in range(3):
            print(f"{STATE_NAMES[i]:>14s}", end="")
            for j in range(3):
                print(f"  {cm[i, j]:12d}", end="")
            print()

        # Per-state metrics
        print(f"\n  Per-State Metrics:")
        print(f"{'State':>14s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}  {'Support':>8s}")
        for s_name, metrics in per_state.items():
            print(f"{s_name:>14s}  {metrics['precision']:8.4f}  {metrics['recall']:8.4f}  "
                  f"{metrics['f1']:8.4f}  {metrics['support']:8d}")

    # Return everything for visualization
    results = {
        "model": model,
        "test_states": test_states,
        "test_obs": test_obs,
        "test_metadata": test_metadata,
        "all_filtered": all_filtered,
        "all_viterbi": all_viterbi,
        "disruption_lags": disruption_lags,
        "recovery_lags": recovery_lags,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_state_metrics": per_state,
    }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    np.savez(os.path.join(results_dir, "evaluation_results.npz"),
             accuracy=accuracy,
             confusion_matrix=cm,
             disruption_lags=np.array(disruption_lags),
             recovery_lags=np.array(recovery_lags),
             pi=model.pi, A=model.A, B=model.B)

    if verbose:
        print(f"\nResults saved to: {results_dir}")

    return results


# Need Counter
from collections import Counter

if __name__ == "__main__":
    run_evaluation()
