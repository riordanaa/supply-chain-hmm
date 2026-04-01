def op_base_stock_all_first(self, all_suppliers):

    # Parameters
    my_supplier_idxs = self.suppliers

    # Base stock policy. on_backorder is not accounted for here because the demand is missed and assumed gone.
    safety_stock_level = self.safety_stock_level
    expected_lead_time = sum([all_suppliers[s].lead_time for s in my_supplier_idxs]) / len(my_supplier_idxs)  # Average lead time of all suppliers
    expected_demand = expected_lead_time*self.customer_demand_mean
    # Calculating on order from all wholsaler suppliers
    total_on_order = 0
    for s in my_supplier_idxs:
        s_idx = self.suppliers.index(s)
        on_orders_all = self.h_orders[(int(-all_suppliers[s].lead_time + 1)):]  # All recent, unfulfilled orders
        on_orders_this_s = [row[s_idx] for row in on_orders_all]
        total_on_order += sum(on_orders_this_s)
    total_on_backorder = sum(self.on_backorder)  # Not used, but here just in case
    current_inventory = self.inventory

    # Determine total_order_amount
    amount_needed = safety_stock_level + expected_demand - total_on_order - current_inventory  ## Currently assumes backlog is dropped/missed
    total_order_amount = max(0, amount_needed)

    # Assign all order to first wholesaler
    order_amounts = [0 for _ in range(self.n_suppliers)]
    order_amounts[0] = total_order_amount

    # Update trackers
    self.orders = order_amounts
    # Update historical trackers
    self.h_orders.append(order_amounts)

    return order_amounts # Integer