"""
Visualization suite for HMM supply chain disruption detection results.

Generates publication-quality plots for the course project report.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from config import (
    STEADY, DISRUPTION, RECOVERY, STATE_NAMES,
    N_OBS, OBS_NAMES, DISRUPTION_ONSET
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
STATE_COLORS = {STEADY: '#2ecc71', DISRUPTION: '#e74c3c', RECOVERY: '#f39c12'}


def plot_hero_figure(test_states, test_obs, all_filtered, all_viterbi,
                     test_metadata, run_idx=0):
    """
    Hero figure: Single test run with ground-truth state bands,
    Viterbi decoded states, and Forward P(Disruption) curve.
    """
    true_s = test_states[run_idx]
    viterbi_s = all_viterbi[run_idx]
    filtered = all_filtered[run_idx]
    meta = test_metadata[run_idx]
    T = len(true_s)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 3]})

    # --- Panel 1: Ground-truth state ---
    ax = axes[0]
    for t in range(T):
        ax.axvspan(t - 0.5, t + 0.5, color=STATE_COLORS[true_s[t]], alpha=0.7)
    ax.set_ylabel("Ground\nTruth", fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title(f"HMM Disruption Detection — Test Run {run_idx}\n"
                 f"(Onset: t={meta['onset']}, Recovery: t={meta.get('recovery_time', 'never')})",
                 fontsize=13, fontweight='bold')

    # --- Panel 2: Viterbi decoded state ---
    ax = axes[1]
    for t in range(T):
        ax.axvspan(t - 0.5, t + 0.5, color=STATE_COLORS[viterbi_s[t]], alpha=0.7)
    ax.set_ylabel("Viterbi\nDecoded", fontsize=10, fontweight='bold')
    ax.set_yticks([])

    # --- Panel 3: Forward probabilities ---
    ax = axes[2]
    t_range = np.arange(T)
    for s in [STEADY, DISRUPTION, RECOVERY]:
        ax.plot(t_range, filtered[:, s], color=STATE_COLORS[s],
                linewidth=2, label=f"P({STATE_NAMES[s]})")

    # Mark onset and recovery
    ax.axvline(x=meta['onset'], color='black', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f"Disruption onset (t={meta['onset']})")
    if meta.get('recovery_time'):
        ax.axvline(x=meta['recovery_time'], color='black', linestyle=':',
                   linewidth=1.5, alpha=0.7,
                   label=f"MN recovers (t={meta['recovery_time']})")

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Period (t)", fontsize=11)
    ax.set_ylabel("Filtered Probability\nP(State | Observations)", fontsize=10, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Legend for state colors
    legend_patches = [mpatches.Patch(color=STATE_COLORS[s], alpha=0.7, label=STATE_NAMES[s])
                      for s in [STEADY, DISRUPTION, RECOVERY]]
    axes[0].legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "hero_figure.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_forward_probabilities(all_filtered, test_states, test_metadata, run_idx=0):
    """Detailed Forward probability evolution for a single run."""
    filtered = all_filtered[run_idx]
    true_s = test_states[run_idx]
    meta = test_metadata[run_idx]
    T = len(true_s)

    fig, ax = plt.subplots(figsize=(12, 5))
    t_range = np.arange(T)

    # Background state colors
    for t in range(T):
        ax.axvspan(t - 0.5, t + 0.5, color=STATE_COLORS[true_s[t]], alpha=0.15)

    for s in [STEADY, DISRUPTION, RECOVERY]:
        ax.plot(t_range, filtered[:, s], color=STATE_COLORS[s],
                linewidth=2.5, label=f"P({STATE_NAMES[s]} | obs_1:t)")

    ax.axvline(x=meta['onset'], color='black', linestyle='--', linewidth=1.5)
    if meta.get('recovery_time'):
        ax.axvline(x=meta['recovery_time'], color='black', linestyle=':',
                   linewidth=1.5)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Period (t)", fontsize=11)
    ax.set_ylabel("Filtered State Probability", fontsize=11)
    ax.set_title("Forward Algorithm: Real-Time State Probability Evolution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "forward_probabilities.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm):
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize for color (but show raw counts)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            pct = 100 * cm_norm[i, j]
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha='center', va='center', fontsize=12, color=text_color,
                    fontweight='bold')

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(STATE_NAMES, fontsize=11)
    ax.set_yticklabels(STATE_NAMES, fontsize=11)
    ax.set_xlabel("Predicted State", fontsize=12)
    ax.set_ylabel("True State", fontsize=12)
    ax.set_title("Viterbi Classification — Confusion Matrix", fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Row-normalized proportion")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_detection_lag_histogram(disruption_lags, recovery_lags):
    """Detection lag distribution across test runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Disruption lags
    ax = axes[0]
    if disruption_lags:
        ax.hist(disruption_lags, bins=range(0, max(disruption_lags) + 2),
                color=STATE_COLORS[DISRUPTION], alpha=0.8, edgecolor='black')
        ax.axvline(x=np.mean(disruption_lags), color='black', linestyle='--',
                   linewidth=2, label=f"Mean: {np.mean(disruption_lags):.1f} weeks")
        ax.legend(fontsize=10)
    ax.set_xlabel("Detection Lag (weeks)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Disruption Detection Lag\n(Forward P(Disruption) > 0.5)", fontsize=12, fontweight='bold')

    # Recovery lags
    ax = axes[1]
    if recovery_lags:
        ax.hist(recovery_lags, bins=range(0, max(recovery_lags) + 2),
                color=STATE_COLORS[RECOVERY], alpha=0.8, edgecolor='black')
        ax.axvline(x=np.mean(recovery_lags), color='black', linestyle='--',
                   linewidth=2, label=f"Mean: {np.mean(recovery_lags):.1f} weeks")
        ax.legend(fontsize=10)
    ax.set_xlabel("Detection Lag (weeks)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Recovery Detection Lag\n(Forward P(Recovery) > 0.5)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "detection_lag_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_trained_matrices(model):
    """Heatmaps of trained A and B matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Transition matrix A
    ax = axes[0]
    im = ax.imshow(model.A, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            text_color = 'white' if model.A[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{model.A[i, j]:.4f}", ha='center', va='center',
                    fontsize=12, color=text_color, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(STATE_NAMES, fontsize=10)
    ax.set_yticklabels(STATE_NAMES, fontsize=10)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title("Transition Matrix A", fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Emission matrix B
    ax = axes[1]
    im = ax.imshow(model.B, cmap='Greens', vmin=0, vmax=1)
    for i in range(3):
        for o in range(model.n_obs):
            text_color = 'white' if model.B[i, o] > 0.5 else 'black'
            ax.text(o, i, f"{model.B[i, o]:.3f}", ha='center', va='center',
                    fontsize=10, color=text_color)
    ax.set_xticks(range(model.n_obs))
    ax.set_yticks(range(3))
    short_obs_names = [f"obs={o}" for o in range(model.n_obs)]
    ax.set_xticklabels(short_obs_names, fontsize=9)
    ax.set_yticklabels(STATE_NAMES, fontsize=10)
    ax.set_xlabel("Observation")
    ax.set_ylabel("State")
    ax.set_title("Emission Matrix B", fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "trained_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_raw_signals(run_idx=0):
    """
    Plot raw DR signals (shipment received, backlog) with state-colored background.
    Uses raw CSV data directly.
    """
    import pandas as pd

    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")

    # Determine which raw run corresponds to test run_idx
    # Test runs start at index n_train
    train_data = np.load(os.path.join(processed_dir, "train.npz"), allow_pickle=True)
    n_train = len(train_data["states"])
    actual_run_id = n_train + run_idx

    df = pd.read_csv(os.path.join(raw_dir, f"run_{actual_run_id:03d}.csv"))
    with open(os.path.join(raw_dir, f"run_{actual_run_id:03d}_meta.json")) as f:
        meta = json.load(f)

    # Assign ground truth for coloring
    from config import PRODUCTION_MAX, RECOVERY_BACKLOG_THRESHOLD
    states = []
    for t in range(len(df)):
        pm = df["mn_production_max"].iloc[t]
        mb = df["mn_backlog"].iloc[t]
        if pm < PRODUCTION_MAX:
            states.append(DISRUPTION)
        elif mb > RECOVERY_BACKLOG_THRESHOLD:
            states.append(RECOVERY)
        else:
            states.append(STEADY)

    T = len(df)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Background state colors for all panels
    for ax in axes:
        for t in range(T):
            ax.axvspan(t - 0.5, t + 0.5, color=STATE_COLORS[states[t]], alpha=0.15)

    # Panel 1: DR Shipment Received
    ax = axes[0]
    ax.plot(df["t"], df["dr_shipment_received"], 'b-', linewidth=1.5)
    ax.set_ylabel("DR Shipment\nReceived", fontsize=10, fontweight='bold')
    ax.axvline(x=meta['onset'], color='black', linestyle='--', linewidth=1)
    if meta.get('recovery_time'):
        ax.axvline(x=meta['recovery_time'], color='black', linestyle=':', linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Raw DR Signals — Run {actual_run_id}\n"
                 f"(Background: Ground-Truth State)", fontsize=13, fontweight='bold')

    # Panel 2: DR Backlog
    ax = axes[1]
    ax.plot(df["t"], df["dr_backlog"], 'r-', linewidth=1.5)
    ax.set_ylabel("DR Backlog\n(from MN)", fontsize=10, fontweight='bold')
    ax.axvline(x=meta['onset'], color='black', linestyle='--', linewidth=1)
    if meta.get('recovery_time'):
        ax.axvline(x=meta['recovery_time'], color='black', linestyle=':', linewidth=1)
    ax.grid(True, alpha=0.3)

    # Panel 3: MN Production Max (the hidden variable)
    ax = axes[2]
    ax.plot(df["t"], df["mn_production_max"], 'k-', linewidth=2)
    ax.set_ylabel("MN Production\nCapacity (HIDDEN)", fontsize=10, fontweight='bold')
    ax.set_xlabel("Period (t)", fontsize=11)
    ax.axvline(x=meta['onset'], color='black', linestyle='--', linewidth=1,
               label=f"Disruption (t={meta['onset']})")
    if meta.get('recovery_time'):
        ax.axvline(x=meta['recovery_time'], color='black', linestyle=':',
                   linewidth=1, label=f"Recovery (t={meta['recovery_time']})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Legend
    legend_patches = [mpatches.Patch(color=STATE_COLORS[s], alpha=0.3, label=STATE_NAMES[s])
                      for s in [STEADY, DISRUPTION, RECOVERY]]
    axes[0].legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "raw_signals.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots(results):
    """Generate all visualization plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nGenerating plots...")

    # Find a good example run (one that has all 3 states and recovered)
    best_idx = 0
    for i, meta in enumerate(results["test_metadata"]):
        if meta.get("recovery_time") is not None:
            best_idx = i
            break

    plot_hero_figure(
        results["test_states"], results["test_obs"],
        results["all_filtered"], results["all_viterbi"],
        results["test_metadata"], run_idx=best_idx
    )

    plot_forward_probabilities(
        results["all_filtered"], results["test_states"],
        results["test_metadata"], run_idx=best_idx
    )

    plot_confusion_matrix(results["confusion_matrix"])

    plot_detection_lag_histogram(
        results["disruption_lags"], results["recovery_lags"]
    )

    plot_trained_matrices(results["model"])

    plot_raw_signals(run_idx=best_idx)

    print(f"\nAll plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    # Load saved results and regenerate plots
    from evaluate import run_evaluation
    results = run_evaluation(verbose=False)
    generate_all_plots(results)
