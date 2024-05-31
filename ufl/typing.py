"""Typing."""

__all__ = ["Self"]

# TODO: Remove this when Python 3.11 is the minimum version
try:
    from typing import Self
except ImportError:
    from typing import Any as Self


# TODO: move this to algorithms?
def cutoff(f):
    """Mark the function as a cutoff so that its children are skipped in traversal."""
    f.cutoff = True
    return f
