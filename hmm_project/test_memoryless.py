"""
Test whether the supply chain system naturally satisfies the memoryless
(Markov) property, using a FIXED-duration disruption (not geometric).

Approach:
1. Run many simulations with a fixed disruption window (e.g., t=50 to t=100).
2. Track the system state at each time step using observable DR signals
   (backlog and shipment received).
3. Measure the "dwell time" in each state (how many consecutive periods
   the system stays in a given state).
4. Test if dwell times follow a Geometric distribution (required for Markov).
5. Also test the first-order Markov assumption: does knowing the state 2
   steps back help predict the next state, beyond what 1 step back tells us?

Output: Statistical test results + plots for the professor.
"""

import sys
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from Simulation import Simulation
from Consumer import Consumer
from instrumented_agents import InstrumentedTransshipper, InstrumentedProducer
from pp_base_stock import pp_base_stock
from ap_proportional import ap_proportional
from op_base_stock_all_first import op_base_stock_all_first
from config import (
    PRODUCTION_MAX, HC_DEMAND, HC_DEMAND_STD, HC_SAFETY_STOCK,
    DR_SAFETY_STOCK, DR_LEAD_TIME, MN_SAFETY_STOCK, MN_LEAD_TIME,
    MN_PROD_LEAD_TIME, SEED_BASE, RECOVERY_BACKLOG_THRESHOLD,
    STEADY, DISRUPTION, RECOVERY, STATE_NAMES
)


# --- Fixed disruption parameters ---
N_RUNS = 200
SIM_PERIODS = 200
DISRUPTION_START = 30
DISRUPTION_END = 50   # Fixed 20-week disruption
DISRUPTION_FACTOR = 0.1  # 90% capacity reduction (matches config)


def make_fixed_disruption(start, end, factor):
    """Create a fixed-duration disruption function."""
    state = {"prod_max_log": []}

    def disruption_fn(sim, t):
        if t == start:
            sim.producers[0].production_max = int(sim.original_production_max[0] * factor)
        elif t == end:
            sim.producers[0].production_max = sim.original_production_max[0]
        state["prod_max_log"].append(sim.producers[0].production_max)

    return disruption_fn, state


def no_policy_change(self, t):
    pass


def create_agents():
    consumers = [
        Consumer(name="HC1", d=HC_DEMAND, dstd=HC_DEMAND_STD, ss=HC_SAFETY_STOCK,
                 suppliers=[0], order_policy_function=op_base_stock_all_first)
    ]
    transhippers = [
        InstrumentedTransshipper(consumers=consumers, name="DR1", suppliers=[0],
                                  customers=[0], ss=DR_SAFETY_STOCK, l=DR_LEAD_TIME,
                                  order_policy_function=op_base_stock_all_first,
                                  allocation_policy_function=ap_proportional)
    ]
    producers = [
        InstrumentedProducer(transhippers=transhippers, name="MN1", ss=MN_SAFETY_STOCK,
                              m=PRODUCTION_MAX, l=MN_LEAD_TIME, pl=MN_PROD_LEAD_TIME,
                              customers=[0], production_policy_function=pp_base_stock,
                              allocation_policy_function=ap_proportional)
    ]
    return consumers, transhippers, producers


def assign_ground_truth(prod_max_log, mn_backlog, original_max):
    """Assign ground-truth hidden states based on MN status."""
    states = []
    for t in range(len(prod_max_log)):
        if prod_max_log[t] < original_max:
            states.append(DISRUPTION)
        elif mn_backlog[t] > RECOVERY_BACKLOG_THRESHOLD:
            states.append(RECOVERY)
        else:
            states.append(STEADY)
    return states


def compute_dwell_times(state_sequence):
    """Compute how many consecutive periods the system stays in each state.
    Returns a dict: {state: [dwell_time_1, dwell_time_2, ...]}"""
    dwells = {STEADY: [], DISRUPTION: [], RECOVERY: []}
    if not state_sequence:
        return dwells

    current_state = state_sequence[0]
    current_count = 1

    for t in range(1, len(state_sequence)):
        if state_sequence[t] == current_state:
            current_count += 1
        else:
            dwells[current_state].append(current_count)
            current_state = state_sequence[t]
            current_count = 1

    dwells[current_state].append(current_count)
    return dwells


def test_geometric_fit(dwell_times, state_name):
    """Test if dwell times follow a geometric distribution.
    Returns (p_value, estimated_p, test_statistic)."""
    if len(dwell_times) < 10:
        return None, None, None

    dwells = np.array(dwell_times)

    # MLE estimate of geometric parameter p = 1/mean
    mean_dwell = dwells.mean()
    estimated_p = 1.0 / mean_dwell

    # Kolmogorov-Smirnov test against geometric distribution
    # scipy.stats.geom uses the "number of trials until first success" parameterization
    ks_stat, p_value = stats.kstest(dwells, 'geom', args=(estimated_p,))

    return p_value, estimated_p, ks_stat


def test_markov_order(state_sequences):
    """Test whether the system is first-order Markov.

    Compare:
    - First-order: P(S_t | S_{t-1})
    - Second-order: P(S_t | S_{t-1}, S_{t-2})

    Use a likelihood ratio test (chi-squared).
    """
    # Count first-order transitions
    first_order = Counter()  # (s_{t-1}, s_t) -> count
    first_order_from = Counter()  # s_{t-1} -> count

    # Count second-order transitions
    second_order = Counter()  # (s_{t-2}, s_{t-1}, s_t) -> count
    second_order_from = Counter()  # (s_{t-2}, s_{t-1}) -> count

    for seq in state_sequences:
        for t in range(2, len(seq)):
            s_prev2 = seq[t - 2]
            s_prev1 = seq[t - 1]
            s_curr = seq[t]

            first_order[(s_prev1, s_curr)] += 1
            first_order_from[s_prev1] += 1

            second_order[(s_prev2, s_prev1, s_curr)] += 1
            second_order_from[(s_prev2, s_prev1)] += 1

    # Log-likelihood for first-order model
    ll_first = 0
    for (s_prev, s_curr), count in first_order.items():
        p = count / first_order_from[s_prev]
        if p > 0:
            ll_first += count * np.log(p)

    # Log-likelihood for second-order model
    ll_second = 0
    for (s_prev2, s_prev1, s_curr), count in second_order.items():
        total = second_order_from[(s_prev2, s_prev1)]
        p = count / total if total > 0 else 0
        if p > 0:
            ll_second += count * np.log(p)

    # Likelihood ratio test statistic: -2 * (ll_restricted - ll_full)
    lr_stat = -2 * (ll_first - ll_second)

    # Degrees of freedom: (N_states - 1) * N_states * (N_states - 1)
    # For 3 states: (3-1) * 3 * (3-1) = 12
    n_states = 3
    df = (n_states - 1) * n_states * (n_states - 1)

    p_value = 1 - stats.chi2.cdf(lr_stat, df)

    return lr_stat, df, p_value, first_order, second_order


def run_test():
    """Main function: run simulations and test memoryless property."""
    print(f"=" * 70)
    print("MEMORYLESS PROPERTY TEST")
    print(f"Fixed disruption: t={DISRUPTION_START} to t={DISRUPTION_END} "
          f"({DISRUPTION_END - DISRUPTION_START} weeks)")
    print(f"Running {N_RUNS} simulations, {SIM_PERIODS} periods each...")
    print(f"=" * 70)

    all_state_sequences = []
    all_dwell_times = {STEADY: [], DISRUPTION: [], RECOVERY: []}

    for run_id in range(N_RUNS):
        np.random.seed(SEED_BASE + run_id)
        consumers, transhippers, producers = create_agents()
        disruption_fn, dstate = make_fixed_disruption(
            DISRUPTION_START, DISRUPTION_END, DISRUPTION_FACTOR
        )

        sim = Simulation(SIM_PERIODS, consumers, transhippers, producers,
                         disruption_fn, no_policy_change)
        sim = sim.run()

        mn = producers[0]
        dr = transhippers[0]

        prod_max_log = dstate["prod_max_log"]
        mn_backlog = [bl[0] for bl in mn.h_backlog[:SIM_PERIODS]]

        states = assign_ground_truth(prod_max_log, mn_backlog, PRODUCTION_MAX)
        all_state_sequences.append(states)

        dwells = compute_dwell_times(states)
        for s in [STEADY, DISRUPTION, RECOVERY]:
            all_dwell_times[s].extend(dwells[s])

        if run_id % 50 == 0:
            print(f"  Completed run {run_id}...")

    print(f"\nAll {N_RUNS} runs completed.")

    # --- Results ---
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Print state distribution
    print(f"\n{'=' * 70}")
    print("STATE DISTRIBUTION ACROSS ALL RUNS")
    print(f"{'=' * 70}")
    all_states_flat = [s for seq in all_state_sequences for s in seq]
    total = len(all_states_flat)
    for s in [STEADY, DISRUPTION, RECOVERY]:
        count = all_states_flat.count(s)
        print(f"  {STATE_NAMES[s]:12s}: {count:6d} periods ({100 * count / total:.1f}%)")

    # Test 1: Dwell time geometric fit
    print(f"\n{'=' * 70}")
    print("TEST 1: DWELL TIME GEOMETRIC DISTRIBUTION FIT")
    print("(Memoryless property requires dwell times ~ Geometric(p))")
    print(f"{'=' * 70}")

    for s in [STEADY, DISRUPTION, RECOVERY]:
        dwells = all_dwell_times[s]
        n_episodes = len(dwells)
        print(f"\n  {STATE_NAMES[s]} state:")
        print(f"    Episodes: {n_episodes}")
        if dwells:
            print(f"    Mean dwell time: {np.mean(dwells):.1f} periods")
            print(f"    Std dwell time:  {np.std(dwells):.1f} periods")
            print(f"    Min: {min(dwells)}, Max: {max(dwells)}")

        p_value, est_p, ks_stat = test_geometric_fit(dwells, STATE_NAMES[s])
        if p_value is not None:
            print(f"    Geometric MLE p: {est_p:.4f} (expected duration = {1/est_p:.1f})")
            print(f"    KS test stat:    {ks_stat:.4f}")
            print(f"    KS p-value:      {p_value:.4f}")
            if p_value < 0.05:
                print(f"    >> REJECTS geometric fit (p < 0.05)")
                print(f"    >> Memoryless property is VIOLATED for this state")
            else:
                print(f"    >> Cannot reject geometric fit (p >= 0.05)")
                print(f"    >> Consistent with memoryless property")
        else:
            print(f"    (Too few episodes to test)")

    # Test 2: First-order vs second-order Markov
    print(f"\n{'=' * 70}")
    print("TEST 2: FIRST-ORDER vs SECOND-ORDER MARKOV (Likelihood Ratio Test)")
    print("(If second-order doesn't improve, system is first-order Markov)")
    print(f"{'=' * 70}")

    lr_stat, df, p_value, fo, so = test_markov_order(all_state_sequences)
    print(f"  LR test statistic: {lr_stat:.2f}")
    print(f"  Degrees of freedom: {df}")
    print(f"  p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  >> Second-order significantly improves over first-order (p < 0.05)")
        print(f"  >> Memoryless property is VIOLATED")
    else:
        print(f"  >> Second-order does NOT significantly improve (p >= 0.05)")
        print(f"  >> Consistent with first-order Markov (memoryless)")

    # Print transition matrices
    print(f"\n  First-order transition counts:")
    for i in range(3):
        row = [fo.get((i, j), 0) for j in range(3)]
        total_from = sum(row)
        probs = [r / total_from if total_from > 0 else 0 for r in row]
        print(f"    {STATE_NAMES[i]:12s} -> " +
              " | ".join(f"{STATE_NAMES[j]}: {probs[j]:.4f}" for j in range(3)))

    # --- PLOTS ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Memoryless Property Analysis\n"
                 f"Fixed Disruption (t={DISRUPTION_START}-{DISRUPTION_END}), "
                 f"{N_RUNS} runs, {SIM_PERIODS} periods each",
                 fontsize=14, fontweight='bold')

    # Plot 1: Dwell time histograms with geometric fit overlay
    for idx, s in enumerate([RECOVERY, DISRUPTION]):
        ax = axes[0, idx]
        dwells = all_dwell_times[s]
        if len(dwells) > 5:
            max_dwell = max(dwells)
            bins = range(1, max_dwell + 2)
            ax.hist(dwells, bins=bins, density=True, alpha=0.7, color=['gold', 'red'][idx],
                    edgecolor='black', label='Observed')

            # Geometric fit overlay
            est_p = 1.0 / np.mean(dwells)
            x = np.arange(1, max_dwell + 1)
            geom_pmf = stats.geom.pmf(x, est_p)
            ax.plot(x, geom_pmf, 'k-o', markersize=4, linewidth=2,
                    label=f'Geometric(p={est_p:.3f})')

            p_val, _, ks = test_geometric_fit(dwells, STATE_NAMES[s])
            verdict = "PASS" if (p_val and p_val >= 0.05) else "FAIL"
            ax.set_title(f"{STATE_NAMES[s]} Dwell Times\n"
                         f"KS p={p_val:.4f} [{verdict}]" if p_val else f"{STATE_NAMES[s]} Dwell Times")
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"Too few {STATE_NAMES[s]}\nepisodes to plot",
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("Dwell Time (periods)")
        ax.set_ylabel("Probability")

    # Plot 2: Example state sequence for a single run
    ax = axes[1, 0]
    example_run = all_state_sequences[0]
    t_range = range(len(example_run))
    colors = {STEADY: 'green', DISRUPTION: 'red', RECOVERY: 'gold'}
    for t in t_range:
        ax.axvspan(t - 0.5, t + 0.5, alpha=0.4, color=colors[example_run[t]])
    ax.axvline(x=DISRUPTION_START, color='black', linestyle='--', linewidth=1.5, label='Disruption start')
    ax.axvline(x=DISRUPTION_END, color='black', linestyle=':', linewidth=1.5, label='Disruption end')
    # Add legend patches
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor='green', alpha=0.4, label='Steady'),
                      Patch(facecolor='red', alpha=0.4, label='Disruption'),
                      Patch(facecolor='gold', alpha=0.4, label='Recovery')]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0], loc='upper right', fontsize=8)
    ax.set_xlabel("Period (t)")
    ax.set_ylabel("State")
    ax.set_title("Example Run: Ground-Truth State Sequence")
    ax.set_xlim(0, SIM_PERIODS)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(STATE_NAMES)

    # Plot 3: Transition probability matrix heatmap
    ax = axes[1, 1]
    trans_matrix = np.zeros((3, 3))
    for i in range(3):
        row = [fo.get((i, j), 0) for j in range(3)]
        total_from = sum(row)
        if total_from > 0:
            trans_matrix[i, :] = [r / total_from for r in row]

    im = ax.imshow(trans_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{trans_matrix[i, j]:.4f}",
                    ha='center', va='center', fontsize=11,
                    color='white' if trans_matrix[i, j] > 0.5 else 'black')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(STATE_NAMES, fontsize=10)
    ax.set_yticklabels(STATE_NAMES, fontsize=10)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title("Empirical Transition Matrix\n(First-Order)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "memoryless_property_test.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    # Summary verdict
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("The memoryless property states that P(leaving state at time t)")
    print("depends only on the current state, not on how long you've been in it.")
    print()
    print("For supply chains with lead times and cumulative backlogs,")
    print("this property is typically VIOLATED because:")
    print("  - Lead times create multi-period memory (order placed 2 weeks ago")
    print("    determines today's shipment)")
    print("  - Backlogs accumulate and take deterministic time to clear")
    print("  - The Recovery phase duration depends on backlog size, not a coin flip")
    print()
    print("This is expected and well-documented in the literature.")
    print("HMMs are still highly effective on such data (as in speech recognition,")
    print("stock markets, etc.), but acknowledging this violation strengthens")
    print("the academic rigor of the project.")
    print()

    # Check if we should recommend geometric encoding
    rec_dwells = all_dwell_times[RECOVERY]
    dis_dwells = all_dwell_times[DISRUPTION]
    rec_geom_pval, _, _ = test_geometric_fit(rec_dwells, "Recovery")
    dis_geom_pval, _, _ = test_geometric_fit(dis_dwells, "Disruption")

    if (rec_geom_pval and rec_geom_pval < 0.05) or (dis_geom_pval and dis_geom_pval < 0.05):
        print("RECOMMENDATION: The dwell time distribution significantly deviates")
        print("from geometric. Encoding disruption duration as Geometric(p) would")
        print("force the memoryless property at the MN level, making the HMM")
        print("theoretically justified. This is recommended for the project.")
    else:
        print("RECOMMENDATION: The data does not strongly reject the geometric fit.")
        print("You may proceed without geometric encoding, but adding it would")
        print("strengthen the theoretical foundation.")

    return all_state_sequences, all_dwell_times


if __name__ == "__main__":
    run_test()
