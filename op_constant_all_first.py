def op_constant_all_first(self, all_suppliers):

    # Constant amount
    total_order_amount = self.customer_demand_mean

    # Assign all order to first wholesaler
    order_amounts = [0 for _ in self.n_suppliers]
    order_amounts[0] = total_order_amount

    # Update trackers
    self.orders = order_amounts
    # Update historical trackers
    self.h_orders.append(order_amounts)

    return order_amounts # Integer