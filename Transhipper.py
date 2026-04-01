##-- Transhipper Agent --##
class Transhipper:

    ## Initialization
    def __init__(self, consumers, name, suppliers, customers, ss, l, order_policy_function, allocation_policy_function):

        # Decision policies
        self.order_policy = order_policy_function
        self.allocation_policy = allocation_policy_function

        # Parameters
        self.name = name
        self.customers = customers  # Customers vector
        self.customer_demand_mean = sum([consumers[i].customer_demand_mean for i in customers])
        self.n_customers = len(customers)
        self.suppliers = suppliers  # Suppliers vector
        self.n_suppliers = len(suppliers)
        self.lead_time = l
        self.safety_stock_level = ss

        # Current trackers
        self.shipments_received = [0 for _ in range(self.n_suppliers)] 
        self.on_backorder = [0 for _ in range(self.n_suppliers)]  # For backorder tracking, basically sum of historic unmet_demand 
        self.inventory = ss
        self.customers_orders = [0 for _ in range(self.n_customers)]
        self.allocations = [0 for _ in range(self.n_customers)]
        self.allocations_queue = [[0 for _ in range(self.n_customers)] for _ in range(self.lead_time)]
        self.unmet_demand = 0
        self.backlog = [0 for _ in range(self.n_customers)]
        self.orders = [0 for _ in range(self.n_suppliers)]
        self.unmet_orders = [0 for _ in range(self.n_suppliers)]
        self.fulfillment_rate = [1 for _ in range(self.n_suppliers)]
        
        # Historical trackers
        self.h_shipment_received = []
        self.h_inventory = []
        self.h_customers_orders = []
        self.h_allocations = []
        self.h_unmet_demand = []
        self.h_backlog = []
        self.h_orders = [[0 for _ in range(self.n_suppliers)] for _ in range(10)]  # Sufficiently greater than lead time, need to initialize here for on_order calculations in order function
        self.h_unmet_orders = []
        self.h_fulfillment_rate = [[1 for _ in range(self.n_suppliers)] for _ in range(10)]  # Sufficiently long for fulfillment rate adjusted order calculation


    def deliver_shipments(self):
        deliveries = self.allocations_queue.pop(0)  # from allocation queue, will be shipment from l weeks ago
        return deliveries

    ## Process-continuation functions
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

    def receive_order(self, consumer_order, consumer_id):
        customer_idx = self.customers.index(consumer_id)
        self.customers_orders[customer_idx] = consumer_order
        
    def allocation_decision(self):
        # Update historical tracker for customer orders here once all orders are in
        self.h_customers_orders.append(self.customers_orders)

        # Update these historical trackers here once all shipments have come in
        self.h_shipment_received.append(self.shipments_received)
        self.h_backlog.append(self.on_backorder)
        self.h_unmet_orders.append(self.unmet_orders)
        self.h_fulfillment_rate.append(self.fulfillment_rate)

        # Allocation decisions
        self.allocation_policy(self)

    def observe_backlog(self):
        for i in range(len(self.backlog)):
            self.backlog[i] = self.backlog[i] + self.customers_orders[i] - self.allocations[i]  # All of length len(customers)

    def send_shipments(self):
        allocations = self.allocations
        self.allocations_queue.append(allocations)  # into allocation queue (in transit)

    def determine_orders(self, all_producers):
        return self.order_policy(self, all_producers)
