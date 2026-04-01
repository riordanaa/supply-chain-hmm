def pp_base_stock(self):

    safety_stock_level = self.safety_stock_level
    customer_demand = self.customer_demand
    expected_demand = self.lead_time*customer_demand
    in_production = sum(self.h_production_decisions[(int(-self.lead_time + 1)):])
    backlog = self.backlog[0]
    amount_needed = safety_stock_level + backlog + expected_demand - in_production 
    production_amount = min(amount_needed, self.production_max)
    self.production_queue.append(production_amount)
    # Update historical trackers
    self.h_production_decisions.append(production_amount)

    return production_amount