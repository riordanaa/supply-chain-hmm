"""
Central configuration for the HMM Supply Chain Disruption Detection project.
All tunable parameters in one place for reproducibility.

Snapshot of the config used to produce the baseline p=0.08 results documented
in report.md Sections 1-7 and archived under hmm_project/p08/results/.
To reproduce this run: copy this file to hmm_project/config.py, then run
`python hmm_project/run_pipeline.py`.
"""

# --- Simulation Parameters ---
N_RUNS = 100
SIM_PERIODS = 80            # weeks per run (enough for pre-disruption, disruption, recovery, return to steady)
DISRUPTION_ONSET = 15       # fixed onset week (gives ~15 weeks of steady-state warmup)
GEOM_P = 0.08               # P(recover each week) => expected disruption duration ~12.5 weeks
PRODUCTION_MAX = 800         # MN max production capacity (high enough for fulfillment + recovery)
DISRUPTION_FACTOR = 0.05     # capacity drops to 5% (40 units/week vs ~200 demand)

# --- Simplified Network Parameters (1 MN, 1 DR, 1 HC) ---
HC_DEMAND = 200              # constant mean demand (units/week)
HC_DEMAND_STD = 10           # demand standard deviation
HC_SAFETY_STOCK = 500        # HC inventory target
DR_SAFETY_STOCK = 500        # DR inventory target (low so disruption is visible)
DR_LEAD_TIME = 2             # shipping lead time (weeks)
MN_SAFETY_STOCK = 0          # MN starts with no buffer (builds up via production)
MN_LEAD_TIME = 2             # shipping lead time (weeks)
MN_PROD_LEAD_TIME = 2        # production lead time (weeks)

# --- Ordering Policy ---
# Using op_base_stock_all_first (base stock, all orders to first supplier)
# This is simpler than fulfillment-rate-adjusted and avoids runaway ordering

# --- Ground-Truth State Definitions ---
# States: STEADY=0, DISRUPTION=1, RECOVERY=2
STEADY = 0
DISRUPTION = 1
RECOVERY = 2
STATE_NAMES = ["Steady", "Disruption", "Recovery"]

RECOVERY_BACKLOG_THRESHOLD = 50  # MN backlog below this => back to steady state

# --- Observation Discretization ---
# Backlog: 2 levels
BACKLOG_HIGH_THRESHOLD = 100     # DR backlog above this = "High"

# Received Shipment: 3 levels (as fraction of pre-disruption baseline)
SHIPMENT_LOW_FRACTION = 0.3      # below this fraction = "Zero/Low"
SHIPMENT_SURGE_FRACTION = 1.3    # above this fraction = "Surge"

N_BACKLOG_LEVELS = 2             # None, High
N_SHIPMENT_LEVELS = 3            # Zero/Low, Normal, Surge
N_OBS = N_BACKLOG_LEVELS * N_SHIPMENT_LEVELS  # M = 6

OBS_NAMES = [
    "None-BL, Zero/Low-Ship",   # 0
    "None-BL, Normal-Ship",     # 1
    "None-BL, Surge-Ship",      # 2
    "High-BL, Zero/Low-Ship",   # 3
    "High-BL, Normal-Ship",     # 4
    "High-BL, Surge-Ship",      # 5
]

# --- Data Split ---
TRAIN_FRACTION = 0.7  # 70 train, 30 test

# --- Lead-Time Adjusted Evaluation ---
LEAD_TIME_SHIFT = 4              # MN_PROD_LEAD_TIME (2) + MN_LEAD_TIME (2)
MICRO_DISRUPTION_THRESHOLD = 5   # disruptions <= 5 weeks are physically unobservable

# --- Warmup Truncation ---
WARMUP_PERIODS = 10          # drop first 10 periods from training sequences
                             # (removes artificial backlog from MN_ss=0 startup,
                             #  keeps 4 clean steady-state weeks before disruption at t=15)

# --- Random Seeds ---
SEED_BASE = 42
