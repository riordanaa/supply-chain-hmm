def ap_proportional(self):

    allocations = [0 for _ in range(self.n_customers)]
    customer_orders = [self.customers_orders[i] + self.backlog[i] for i in range(len(self.customers))]  ## NEW: Customers orders plus backlog
    total_orders_and_backlog = sum(customer_orders) + sum(self.backlog)
    total_supply = self.inventory
    if total_supply < sum(customer_orders):
        for i in range(len(self.customers)):
            allocations[i] = int(min(customer_orders[i] * total_supply / max(total_orders_and_backlog, 1), customer_orders[i]))  # int() rounds down, second max function is a work around
            self.inventory = self.inventory - allocations[i]  # Remove drugs from inventory
    else:
        for i in range(len(self.customers)):
            allocations[i] = customer_orders[i]
            self.inventory = self.inventory - allocations[i]
    self.allocations = allocations
    # Update historical trackers
    self.h_allocations.append(allocations)
    