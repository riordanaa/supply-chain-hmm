##-- Producer Agent --##
class Producer:

    ## Initialization
    def __init__(self, transhippers, name, ss, m, l, pl, customers, production_policy_function, allocation_policy_function):

        # Decision policies
        self.production_policy = production_policy_function
        self.allocation_policy = allocation_policy_function

        # Parameters
        self.name = name
        self.inventory = ss
        self.production_max = m
        self.lead_time = l  # Shipping lead time
        self.production_lead_time = pl # Production lead time
        self.customers = customers
        self.customer_demand = sum([transhippers[i].customer_demand_mean for i in customers])
        self.n_customers = len(customers)
        self.safety_stock_level = ss

        # Current trackers
        self.production_observed = []
        self.customers_orders = [0 for _ in range(self.n_customers)]
        self.production_queue = [0 for _ in range(self.production_lead_time)]
        self.allocations = [0 for _ in range(self.n_customers)]
        self.allocations_queue = [[0 for _ in range(self.n_customers)] for _ in range(self.lead_time)]
        self.backlog = [0] * len(customers)

        # Historical trackers
        self.h_production_decisions = []
        self.h_production_observed = []
        self.h_allocations = []
        self.h_inventory = []
        self.h_customers_orders = []
        self.h_backlog = []

    ## Process-continuation functions
    def observe_production(self):
        prod = self.production_queue.pop(0)
        self.production_observed = prod
        self.inventory = self.inventory + self.production_observed
        # Update historical trackers
        self.h_production_observed.append(prod)
        self.h_inventory.append(self.inventory)

    def receive_order(self, transhipper_order, transhipper_id):
        customer_idx = self.customers.index(transhipper_id)
        self.customers_orders[customer_idx] = transhipper_order

    def allocation_decision(self):
        # Update historical trackers here once all orders are in
        self.h_customers_orders.append(self.customers_orders)
        self.allocation_policy(self)
        
    def observe_backlog(self):
        for i in range(len(self.backlog)):
            self.backlog[i] = self.backlog[i] + self.customers_orders[i] - self.allocations[i]

    def send_shipments(self):
        allocations = self.allocations
        self.allocations_queue.append(allocations)  # into allocation queue (in transit)

    def deliver_shipments(self):
        deliveries = self.allocations_queue.pop(0)  # from allocation queue, will be shipment from l weeks ago
        return deliveries
    
    def production_decision(self):
        self.production_policy(self)
