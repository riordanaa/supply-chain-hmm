"""
Compatibility shim: `Wholesaler` was renamed to `Transhipper`.
This module preserves the old import path for backwards compatibility.
"""

from Transhipper import Transhipper

# Backwards-compatible alias
Wholesaler = Transhipper