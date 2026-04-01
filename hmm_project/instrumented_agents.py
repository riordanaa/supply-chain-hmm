"""
Instrumented subclasses of Transhipper and Producer that fix copy-semantics
bugs in history tracking and add missing history recording.

Issues in original code:
1. Transhipper.allocation_decision() appends mutable lists without .copy(),
   so all h_backlog/h_shipment_received entries share the same reference.
2. Transhipper never records h_inventory.
3. Producer.observe_backlog() never appends to h_backlog.
4. Producer.allocation_decision() appends customers_orders without .copy().

These subclasses fix all issues without modifying the original files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Transhipper import Transhipper
from Producer import Producer


class InstrumentedTransshipper(Transhipper):
    """Transhipper with copy-safe history tracking and inventory recording."""

    def allocation_decision(self):
        # Record history with copies (fixes shared-reference bug)
        self.h_customers_orders.append(list(self.customers_orders))
        self.h_shipment_received.append(list(self.shipments_received))
        self.h_backlog.append(list(self.on_backorder))
        self.h_unmet_orders.append(list(self.unmet_orders))
        self.h_fulfillment_rate.append(list(self.fulfillment_rate))

        # Record inventory (missing from original)
        self.h_inventory.append(self.inventory)

        # Call allocation policy (same as original)
        self.allocation_policy(self)


class InstrumentedProducer(Producer):
    """Producer with backlog history recording and copy-safe orders."""

    def allocation_decision(self):
        # Copy-safe customer orders
        self.h_customers_orders.append(list(self.customers_orders))
        # Call allocation policy (same as original)
        self.allocation_policy(self)

    def observe_backlog(self):
        # Call original logic
        super().observe_backlog()
        # Record backlog (missing from original)
        self.h_backlog.append(list(self.backlog))
