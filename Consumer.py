# Packages
import numpy as np

##-- Consumer Agent --##
class Consumer:

    ## Initialization
    def __init__(self, name, d, dstd, ss, suppliers, order_policy_function):

        # Decision policies
        self.order_policy = order_policy_function

        # Parameters
        self.name = name
        self.customer_demand_mean = d  # Demand for all inventories, assumes all drug products inventory are in the same units
        self.demand_std = dstd  # Standard deviation of demand
        self.suppliers = suppliers
        self.n_suppliers = len(suppliers)
        self.safety_stock_level = ss

        # Current trackers
        self.observed_demand = 0
        self.inventory = ss
        self.runway = 0
        self.unmet_demand = 0
        self.shipments_received = [0 for _ in range(self.n_suppliers)]  # Tracked for each supplier
        self.on_backorder = [0 for _ in range(self.n_suppliers)]  # For backorder tracking, basically sum of historic unmet_demand 
        self.orders = [0 for _ in range(self.n_suppliers)]
        self.unmet_orders = [0 for _ in range(self.n_suppliers)]
        self.fulfillment_rate = [1] * self.n_suppliers

        # Historical trackers
        self.h_inventory = []
        self.h_runway = []
        self.h_orders = [[0 for _ in range(self.n_suppliers)] for _ in range(10)]  # Sufficiently greater than lead time, need to initialize here for on_order calculations in order function
        self.h_shipment_received = []
        self.h_observed_demand = []
        self.h_unmet_demand = []
        self.h_backlog = []  # To incorporate backlog
        self.h_unmet_orders = []
        self.h_fulfillment_rate = [[1 for _ in range(self.n_suppliers)] for _ in range(10)]  # Sufficiently long for fulfillment rate adjusted order calculation

    ## Simple process-continuation functions
    def receive_shipment(self, total_received, supplier, supplier_idx):

        # Idx needed to index any of the arrays in "trackers" above
        my_supplier_idx = self.suppliers.index(supplier_idx)

        self.shipments_received[my_supplier_idx] = total_received
        self.inventory = self.inventory + total_received

        ## Fulfillment rate
        this_order = self.h_orders[int(-supplier.lead_time)][my_supplier_idx]  # What was received from this order, can use this to track fulfillment rate
        fulfillment_rate = total_received / this_order if this_order != 0 else 1  # If statement for if order is 1, fulfillment rate becomes 1
        self.fulfillment_rate[my_supplier_idx] = fulfillment_rate

        ## Track backlog
        remove_from_backorder = min(total_received, self.on_backorder[my_supplier_idx])
        self.on_backorder[my_supplier_idx] = self.on_backorder[my_supplier_idx] - remove_from_backorder
        remainder_received = total_received - remove_from_backorder
        unmet_order = this_order - remainder_received
        self.unmet_orders[my_supplier_idx] = unmet_order
        self.on_backorder[my_supplier_idx] = self.on_backorder[my_supplier_idx] + unmet_order

    def observe_demand(self):

        self.observed_demand = int(np.random.normal(loc=self.customer_demand_mean, scale=self.demand_std, size=None))
        # Update historical trackers
        self.h_observed_demand.append(self.observed_demand)

        # Update these historical trackers here once all shipments have come in
        self.h_shipment_received.append(self.shipments_received.copy())  # Need to use copy to avoid referencing
        self.h_backlog.append(self.on_backorder.copy())
        self.h_unmet_orders.append(self.unmet_orders.copy())
        self.h_fulfillment_rate.append(self.fulfillment_rate.copy())

    def serve_demand(self):
        demand = self.observed_demand
        amount_used = min(self.inventory, demand)
        unmet_demand = demand - amount_used
        self.inventory = self.inventory - amount_used
        self.unmet_demand = unmet_demand  # Mark unmet demand
        # Update runway
        r = str(round(self.inventory / self.customer_demand_mean, 2)) # Current inventory / demand per week (utilization)
        self.runway = r
        # Update historical trackers
        self.h_runway.append(self.runway)
        self.h_unmet_demand.append(unmet_demand)
        self.h_inventory.append(self.inventory)

    def determine_orders(self, all_transhippers):
        return self.order_policy(self, all_transhippers)
