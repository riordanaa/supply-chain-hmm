"""
Compatibility shim: `Manufacturer` was renamed to `Producer`.
This module preserves the old import path for backwards compatibility.
"""

from Producer import Producer

# Backwards-compatible alias
Manufacturer = Producer