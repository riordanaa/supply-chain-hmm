"""
Data generation wrapper: configures a simplified 1-MN / 1-DR / 1-HC network,
introduces a geometric-duration disruption, runs N replications, and saves
raw data to CSV files.

Does NOT modify any original simulator files.
"""

import sys
import os
import json
import random
import numpy as np
import pandas as pd

# Add parent directory to path to import simulator classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Simulation import Simulation
from Consumer import Consumer
from instrumented_agents import InstrumentedTransshipper, InstrumentedProducer

# Import policies
from pp_base_stock import pp_base_stock
from ap_proportional import ap_proportional
from op_base_stock_all_first import op_base_stock_all_first

# Import project config
from config import (
    N_RUNS, SIM_PERIODS, DISRUPTION_ONSET, GEOM_P,
    PRODUCTION_MAX, DISRUPTION_FACTOR,
    HC_DEMAND, HC_DEMAND_STD, HC_SAFETY_STOCK,
    DR_SAFETY_STOCK, DR_LEAD_TIME,
    MN_SAFETY_STOCK, MN_LEAD_TIME, MN_PROD_LEAD_TIME,
    SEED_BASE
)


def make_disruption_function(onset, geom_p, factor, rng):
    """
    Create a disruption function with geometric recovery duration.

    The disruption triggers at `onset` and recovers each subsequent week
    with probability `geom_p` (memoryless / geometric distribution).

    Returns (disruption_fn, state_dict) where state_dict tracks metadata.
    """
    state = {
        "disrupted": False,
        "recovered": False,
        "recovery_time": None,
        "prod_max_log": [],
    }

    def disruption_fn(sim, t):
        if t == onset and not state["disrupted"]:
            sim.producers[0].production_max = int(sim.original_production_max[0] * factor)
            state["disrupted"] = True
        elif state["disrupted"] and not state["recovered"] and t > onset:
            if rng.random() < geom_p:
                sim.producers[0].production_max = sim.original_production_max[0]
                state["recovered"] = True
                state["recovery_time"] = t

        # Always log current production_max
        state["prod_max_log"].append(sim.producers[0].production_max)

    return disruption_fn, state


def no_policy_change(self, t):
    """No mid-simulation policy changes."""
    pass


def create_agents():
    """Create fresh 1-HC, 1-DR, 1-MN agent instances."""
    consumers = [
        Consumer(
            name="HC1",
            d=HC_DEMAND,
            dstd=HC_DEMAND_STD,
            ss=HC_SAFETY_STOCK,
            suppliers=[0],
            order_policy_function=op_base_stock_all_first,
        )
    ]

    transhippers = [
        InstrumentedTransshipper(
            consumers=consumers,
            name="DR1",
            suppliers=[0],
            customers=[0],
            ss=DR_SAFETY_STOCK,
            l=DR_LEAD_TIME,
            order_policy_function=op_base_stock_all_first,
            allocation_policy_function=ap_proportional,
        )
    ]

    producers = [
        InstrumentedProducer(
            transhippers=transhippers,
            name="MN1",
            ss=MN_SAFETY_STOCK,
            m=PRODUCTION_MAX,
            l=MN_LEAD_TIME,
            pl=MN_PROD_LEAD_TIME,
            customers=[0],
            production_policy_function=pp_base_stock,
            allocation_policy_function=ap_proportional,
        )
    ]

    return consumers, transhippers, producers


def run_single(run_id, verbose=False):
    """Run a single simulation replication and return data + metadata."""
    # Set numpy seed for demand stochasticity
    np.random.seed(SEED_BASE + run_id)

    # Separate RNG for geometric disruption draws
    disruption_rng = random.Random(SEED_BASE + 10000 + run_id)

    # Create fresh agents
    consumers, transhippers, producers = create_agents()

    # Create disruption function
    disruption_fn, disruption_state = make_disruption_function(
        onset=DISRUPTION_ONSET,
        geom_p=GEOM_P,
        factor=DISRUPTION_FACTOR,
        rng=disruption_rng,
    )

    # Run simulation
    sim = Simulation(
        SIM_PERIODS, consumers, transhippers, producers,
        disruption_fn, no_policy_change
    )
    sim = sim.run()

    # Extract data
    mn = producers[0]
    dr = transhippers[0]

    n = SIM_PERIODS
    data = {
        "t": list(range(n)),
        "mn_production_max": disruption_state["prod_max_log"],
        "mn_inventory": mn.h_inventory[:n],
        "mn_backlog": [bl[0] for bl in mn.h_backlog[:n]],
        "mn_production": mn.h_production_decisions[:n],
        "mn_allocation": [a[0] for a in mn.h_allocations[:n]],
        "dr_shipment_received": [sr[0] for sr in dr.h_shipment_received[:n]],
        "dr_inventory": dr.h_inventory[:n],
        "dr_backlog": [bl[0] for bl in dr.h_backlog[:n]],
        "dr_orders_from_hc": [co[0] for co in dr.h_customers_orders[:n]],
    }

    # Validate lengths
    for key, vals in data.items():
        assert len(vals) == n, f"Run {run_id}: {key} has length {len(vals)}, expected {n}"

    metadata = {
        "run_id": run_id,
        "onset": DISRUPTION_ONSET,
        "recovery_time": disruption_state["recovery_time"],
        "geom_p": GEOM_P,
        "disruption_factor": DISRUPTION_FACTOR,
        "seed_numpy": SEED_BASE + run_id,
        "seed_disruption": SEED_BASE + 10000 + run_id,
        "sim_periods": SIM_PERIODS,
    }

    if verbose:
        rec = disruption_state["recovery_time"]
        dur = (rec - DISRUPTION_ONSET) if rec else "never"
        print(f"  Run {run_id:3d}: disruption duration = {dur} weeks")

    return pd.DataFrame(data), metadata


def run_all(n_runs=None, verbose=True):
    """Run all replications and save to disk."""
    if n_runs is None:
        n_runs = N_RUNS

    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Running {n_runs} simulation replications ({SIM_PERIODS} periods each)...")

    all_metadata = []
    for run_id in range(n_runs):
        df, meta = run_single(run_id, verbose=verbose)

        # Save CSV
        csv_path = os.path.join(raw_dir, f"run_{run_id:03d}.csv")
        df.to_csv(csv_path, index=False)

        # Save metadata
        meta_path = os.path.join(raw_dir, f"run_{run_id:03d}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        all_metadata.append(meta)

    # Summary
    recovery_times = [m["recovery_time"] for m in all_metadata if m["recovery_time"] is not None]
    never_recovered = sum(1 for m in all_metadata if m["recovery_time"] is None)
    if recovery_times:
        durations = [rt - DISRUPTION_ONSET for rt in recovery_times]
        print(f"\nDisruption duration stats (N={len(durations)}):")
        print(f"  Mean: {np.mean(durations):.1f} weeks")
        print(f"  Median: {np.median(durations):.1f} weeks")
        print(f"  Min: {min(durations)}, Max: {max(durations)}")
    if never_recovered:
        print(f"  {never_recovered} runs never recovered (disruption lasted entire simulation)")

    print(f"\nData saved to: {raw_dir}")
    return all_metadata


if __name__ == "__main__":
    run_all()
