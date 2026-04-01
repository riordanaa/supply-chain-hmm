"""
Compatibility shim: `Hospital` was renamed to `Consumer`.
This module preserves the old import path for backwards compatibility.
"""

from Consumer import Consumer

# Backwards-compatible alias
Hospital = Consumer