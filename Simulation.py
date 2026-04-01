##-- Import Classes --##
from Consumer import Consumer
from Transhipper import Transhipper
from Producer import Producer


##-- Class --##
class Simulation():

    ## Initialization
    def __init__(self, sim_periods, consumers, transhippers, producers, disruption_function, change_decision_policies):

        ## SUPPLY CHAIN SETUP
        # ASSUMPTIONS:
        # 1. Only one product is tracked in this system.
        # 2. Any agent can have any number of suppliers and customers

        # Agent initialization
        self.sim_periods = sim_periods  # Number of periods that the simulation runs for
        self.consumers = consumers
        self.transhippers = transhippers
        self.producers = producers

        # Changes that can occur mid simulation
        self.disruption_function = disruption_function
        self.change_decision_policies = change_decision_policies

        ## PARAMETERS
        # Set sizes
        self.n_consumers = len(self.consumers)
        self.n_transhippers = len(self.transhippers)
        self.n_producers = len(self.producers)

        # Initialization of necessary parameters
        self.original_production_max = [self.producers[i].production_max for i in range(self.n_producers)]  # Needed for disruption


    ## Functions
    def enable_disruption(self, t):
        self.disruption_function(self, t)

    def enable_change_decision_policies(self, t):
        self.change_decision_policies(self, t)


    # Simulation runner function (DO NOT CHANGE THIS FUNCTION)
    def run(self):

        print("Starting simulation.")

        ## Run simulation for # sim_periods
        for t in range(self.sim_periods):


            ## Supply Chain Flow:
            ## 1. Change production capacities according to enable_disruption()
            self.enable_disruption(t)


            ## 2. Consumer Actions
            # Consumers receive shipments in groups by transhipper
            for w in range(self.n_transhippers):
                transhipper = self.transhippers[w] # Transhipper info
                shipments = transhipper.deliver_shipments()  # For consumer shipment deliveries

                for c in transhipper.customers:
                    # Consumer receives shipment (transhipper delivers it)
                    c_idx = transhipper.customers.index(c)
                    self.consumers[c].receive_shipment(shipments[c_idx], transhipper, w)  # Need transhipper argument to get lead time

            # Then consumers take their own actions one at a time
            for c in range(self.n_consumers):

                # Consumer observes demand
                self.consumers[c].observe_demand()

                # Consumer serves demand
                self.consumers[c].serve_demand()

                # Consumer determines order amounts for all suppliers
                orders_for_transhippers = self.consumers[c].determine_orders(self.transhippers)  

                # Consumers submit orders to its suppliers
                for w in self.consumers[c].suppliers:
                    w_idx = self.consumers[c].suppliers.index(w)
                    # Transhipper received order here
                    self.transhippers[w].receive_order(orders_for_transhippers[w_idx], c)


            ## 3. Transhipper Actions
            ## Transhippers receive shipments in groups by producer
            for m in range(self.n_producers):
                producer = self.producers[m] # Producer info
                shipments = producer.deliver_shipments()

                for w in producer.customers:
                    # Transhipper receives shipment (producer delivers it)
                    w_idx = producer.customers.index(w)
                    self.transhippers[w].receive_shipment(shipments[w_idx], producer, m)

            # Then transhippers take their own actions one at a time
            for w in range(self.n_transhippers):
                    
                # Transhipper makes allocation decision
                self.transhippers[w].allocation_decision()

                # Transhipper observes the backlog (if any)
                self.transhippers[w].observe_backlog()

                # Transhipper sends shipments into transit
                self.transhippers[w].send_shipments()

                # Transhippers determine order amounds for all suppliers
                orders_for_producers = self.transhippers[w].determine_orders(self.producers) 

                # Transhipper submits orders to producers
                for m in self.transhippers[w].suppliers:
                    m_idx = self.transhippers[w].suppliers.index(m)
                    # Transhipper received order here
                    self.producers[m].receive_order(orders_for_producers[m_idx], w)

            ## 4. Producer Actions
            for m in range(self.n_producers):

                # Producer observes production (after lead time)
                self.producers[m].observe_production()

                # Producer decides allocation to transhippers
                self.producers[m].allocation_decision()

                # Producer observes backlog
                self.producers[m].observe_backlog()

                # Producer sends shipments (adds to shipment queue)
                self.producers[m].send_shipments()

                # Producer decides next production amount
                self.producers[m].production_decision()
            

            print(f"Period: {t+1} done.")


        print('Simulation done.')

        return self