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
    N_OBS, DISRUPTION_ONSET, LEAD_TIME_SHIFT, MICRO_DISRUPTION_THRESHOLD
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


def compute_detection_lag(filtered_probs, true_states, target_state, threshold=0.5,
                          search_after=0):
    """
    Compute how many periods after a state first appears in ground truth
    until the Forward algorithm pushes P(state) above threshold.

    search_after: only begin scanning for the target state at or after this
                  time index.  Pass DISRUPTION_ONSET when measuring the
                  Recovery lag so the startup transient (MN begins with zero
                  inventory, briefly triggering a Recovery label near t=0)
                  is excluded.

    Returns lag (int) or None if threshold never exceeded.
    """
    # Find first period at or after search_after where target state appears
    first_true = None
    for t in range(search_after, len(true_states)):
        if true_states[t] == target_state:
            first_true = t
            break

    if first_true is None:
        return None  # State never appears (after search_after)

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


def compute_shifted_viterbi_metrics(viterbi_paths, true_state_sequences, shift=LEAD_TIME_SHIFT):
    """
    Compute Viterbi metrics with ground-truth labels shifted forward by `shift` periods.

    This accounts for the physical lead-time: the DR cannot observe any MN event
    until `shift` weeks later, so comparing Viterbi output at time t against the
    ground truth at time (t - shift) gives a fairer evaluation.

    Implementation: truncate the first `shift` periods from ground truth and the
    last `shift` periods from predictions, so they align.
    """
    shifted_true_list = []
    shifted_pred_list = []

    for true_s, pred_s in zip(true_state_sequences, viterbi_paths):
        true_arr = np.asarray(true_s)
        pred_arr = np.asarray(pred_s)
        T = min(len(true_arr), len(pred_arr))

        if T <= shift:
            continue

        # At time t, the DR's prediction should be compared against
        # the MN's state at time (t - shift)
        shifted_true = true_arr[:T - shift]   # ground truth from t=0 to T-shift-1
        shifted_pred = pred_arr[shift:]        # predictions from t=shift to T-1
        shifted_true_list.append(shifted_true)
        shifted_pred_list.append(shifted_pred)

    return compute_viterbi_metrics(shifted_pred_list, shifted_true_list)


def compute_filtered_accuracy(viterbi_paths, true_state_sequences, test_metadata,
                               max_weeks=MICRO_DISRUPTION_THRESHOLD):
    """
    Compute Viterbi accuracy excluding runs where the disruption lasted <= max_weeks.

    These micro-disruptions are physically absorbed by safety stock buffers
    and leave no observable trace for the DR to detect.
    """
    filtered_pred = []
    filtered_true = []
    n_excluded = 0

    for pred_s, true_s, meta in zip(viterbi_paths, true_state_sequences, test_metadata):
        onset = meta["onset"]
        recovery = meta.get("recovery_time")

        # Compute disruption duration
        if recovery is not None:
            duration = recovery - onset
        else:
            duration = float('inf')  # never recovered — include this run

        if duration <= max_weeks:
            n_excluded += 1
            continue

        filtered_pred.append(pred_s)
        filtered_true.append(true_s)

    if not filtered_pred:
        return None, None, None, 0, n_excluded

    accuracy, cm, per_state = compute_viterbi_metrics(filtered_pred, filtered_true)
    n_included = len(filtered_pred)

    return accuracy, cm, per_state, n_included, n_excluded


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

        # Detection lag for recovery (only post-disruption onset; skip startup transient)
        lag_r = compute_detection_lag(filtered, true_s, RECOVERY, threshold=0.5,
                                      search_after=DISRUPTION_ONSET)
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

    # --- Lead-Time Adjusted Metrics ---
    adj_accuracy, adj_cm, adj_per_state = compute_shifted_viterbi_metrics(
        all_viterbi, test_states, shift=LEAD_TIME_SHIFT
    )

    # --- Filtered Accuracy (excluding micro-disruptions) ---
    filt_accuracy, filt_cm, filt_per_state, n_included, n_excluded = compute_filtered_accuracy(
        all_viterbi, test_states, test_metadata, max_weeks=MICRO_DISRUPTION_THRESHOLD
    )

    if verbose:
        print(f"\n--- Lead-Time Adjusted Performance (ground truth shifted +{LEAD_TIME_SHIFT} weeks) ---")
        print(f"  Adjusted accuracy: {adj_accuracy:.4f} ({adj_accuracy*100:.1f}%)")

        print(f"\n  Adjusted Confusion Matrix (rows=true, cols=predicted):")
        print(f"{'':>14s}", end="")
        for j in range(3):
            print(f"  {STATE_NAMES[j]:>12s}", end="")
        print()
        for i in range(3):
            print(f"{STATE_NAMES[i]:>14s}", end="")
            for j in range(3):
                print(f"  {adj_cm[i, j]:12d}", end="")
            print()

        print(f"\n  Adjusted Per-State Metrics:")
        print(f"{'State':>14s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}  {'Support':>8s}")
        for s_name, metrics in adj_per_state.items():
            print(f"{s_name:>14s}  {metrics['precision']:8.4f}  {metrics['recall']:8.4f}  "
                  f"{metrics['f1']:8.4f}  {metrics['support']:8d}")

        print(f"\n--- Filtered Accuracy (excluding disruptions <= {MICRO_DISRUPTION_THRESHOLD} weeks) ---")
        print(f"  Runs excluded: {n_excluded}, Runs included: {n_included}")
        if filt_accuracy is not None:
            print(f"  Filtered accuracy: {filt_accuracy:.4f} ({filt_accuracy*100:.1f}%)")
            print(f"\n  Filtered Per-State Metrics:")
            print(f"{'State':>14s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}  {'Support':>8s}")
            for s_name, metrics in filt_per_state.items():
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
        "adj_accuracy": adj_accuracy,
        "adj_confusion_matrix": adj_cm,
        "adj_per_state_metrics": adj_per_state,
        "filt_accuracy": filt_accuracy,
        "filt_confusion_matrix": filt_cm,
        "filt_per_state_metrics": filt_per_state,
        "filt_n_included": n_included,
        "filt_n_excluded": n_excluded,
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
