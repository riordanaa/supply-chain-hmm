def op_constant_even_split(self, all_suppliers):

    # Constant amount
    total_order_amount = self.customer_demand_mean

    # Split order between all suppliers
    order_amounts = [0 for _ in self.n_suppliers]
    for i in range(self.n_suppliers):
        order_amounts[i] = int(total_order_amount / self.n_suppliers)

    # Update trackers
    self.orders = order_amounts
    # Update historical trackers
    self.h_orders.append(order_amounts)

    return order_amounts # Integer