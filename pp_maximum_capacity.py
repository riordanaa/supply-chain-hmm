def pp_maximum_capacity(self):

    amount = self.production_max
    production_amount = amount
    self.production_queue.append(amount)
    self.h_production_decisions.append(amount)

    return production_amount