## Supply Chain Simulator
## Developer: Noah Chicoine
## Last Updated: 06/25/2025

##-- Code Dependence --##
## -- main.py
##     --Simulation()
##          -- Manufacturer.py
##              -- pp, ap functions
##          -- Wholesaler.py
##              -- op, ap functions
##          -- Hospital.py
##              -- op, pp functions


##-- Import Packages --##
import pandas as pd
import tkinter as tk


##-- Import Classes --##
from Simulation import Simulation
from Consumer import Consumer
from Transhipper import Transhipper
from Producer import Producer


##-- Import Decision Policy Function --##
## MAKE SURE ALL THAT YOU PLAN TO INCLUDE ARE IMPORTED, IMPORT ALL JUST TO BE SAVE
# Production policies (pp)
from pp_base_stock import pp_base_stock
from pp_maximum_capacity import pp_maximum_capacity
# Allocation policies (ap)
from ap_proportional import ap_proportional
# Ordering policies (op)
from op_base_stock_all_first import op_base_stock_all_first
from op_base_stock_even_split import op_base_stock_even_split
from op_constant_all_first import op_constant_all_first
from op_constant_even_split import op_constant_even_split
from op_base_stock_fr_all_first import op_base_stock_fr_all_first


##-- Functions (can change if needed)--##
def get_results(simulation):

    ## Output
    consumers = simulation.consumers

    ## Export hospital data
    t = range(simulation.sim_periods)
    df = pd.DataFrame()
    for c in consumers:
        df[f"{c.name} shipments received"] = c.h_shipment_received
        df[f"{c.name} demand"] = c.h_observed_demand
        df[f"{c.name} unfulfilled demand"] = c.h_unmet_demand
        df[f"{c.name} inventory"] = c.h_inventory
        df[f"{c.name} runway"] = c.h_runway
        df[f"{c.name} order amounts"] = c.h_orders[-simulation.sim_periods:] # To get rid of first couple of 0s programmed in
        df[f"{c.name} unmet order"] = c.h_unmet_orders
        df[f"{c.name} backlog"] = c.h_backlog
        df[f"{c.name} fulfillment rate"] = c.h_fulfillment_rate[-simulation.sim_periods:]
        
    df.to_excel("consumer_data.xlsx", index=False)  # Saves in current folder
    print("Saved results to Excel.")

    # Return DataFrame for programmatic use (e.g., web app downloads)
    return df


##-- main --##
if __name__ == '__main__':

    ##-- BEGIN: EDIT HERE TO SET UP SIMULATION --##
    sim_periods = 300 

    ## NOTE: Make sure supplier indexes and customer indexes line up across agents!!! This will be made more user-friendly in a future release
    consumers = [Consumer(name="H1", d=150, dstd=10, ss=1000, suppliers=[0, 1], order_policy_function=op_base_stock_fr_all_first),
                 Consumer(name="H2", d=170, dstd=10, ss=1000, suppliers=[0, 2], order_policy_function=op_base_stock_fr_all_first),
                 Consumer(name="H3", d=200, dstd=10, ss=1000, suppliers=[0, 1], order_policy_function=op_base_stock_fr_all_first),
                 Consumer(name="H4", d=50,  dstd=10, ss=1000, suppliers=[0],    order_policy_function=op_base_stock_fr_all_first),
                 Consumer(name="H5", d=200, dstd=10, ss=1000, suppliers=[1],    order_policy_function=op_base_stock_fr_all_first),
                 Consumer(name="H6", d=200, dstd=10, ss=1000, suppliers=[2],    order_policy_function=op_base_stock_fr_all_first)]

    transhippers = [Transhipper(consumers=consumers, name="WS1", suppliers=[0, 1], customers=[0, 1, 2, 3], ss=8000, l=2, order_policy_function=op_base_stock_fr_all_first, allocation_policy_function=ap_proportional),
                   Transhipper(consumers=consumers, name="WS2", suppliers=[0],    customers=[0, 2, 4],    ss=8000, l=2, order_policy_function=op_base_stock_fr_all_first, allocation_policy_function=ap_proportional),
                   Transhipper(consumers=consumers, name="WS3", suppliers=[1],    customers=[1, 5],       ss=8000, l=2, order_policy_function=op_base_stock_fr_all_first, allocation_policy_function=ap_proportional)]
    
    producers = [Producer(transhippers=transhippers, name="MN1", ss=10000, m=800, l=2, pl=2, customers=[0, 1], production_policy_function=pp_base_stock, allocation_policy_function=ap_proportional),
                     Producer(transhippers=transhippers, name="MN2", ss=10000, m=800, l=2, pl=2, customers=[0, 2], production_policy_function=pp_base_stock, allocation_policy_function=ap_proportional)]
    
    
    def disruption_function(self, t):
        # Disruption profile for a simulation
        # Always define this function even if there is no disruption

        # Disruption at MN 1, change their production_max for 50 periods
        if t == 100:
            self.producers[0].production_max = self.producers[0].production_max * 0.2 # Production capacity down to 20%
        if t == 151:
            self.producers[0].production_max = self.original_production_max[0]  # Production capacity returns to normal


    def change_decision_policies(self, t):
        # Function for changing any agents' decision policies mid-way through simulation
        # Always define this function even if there is no change in any policies
        pass

    ##-- END: EDIT HERE TO SET UP SIMULATION --##


    ##-- Run simulation and collect results (DO NOT EDIT) --##
    simulation = Simulation(sim_periods, consumers, transhippers, producers, disruption_function, change_decision_policies)
    simulation = simulation.run()
    df_results = get_results(simulation)

    