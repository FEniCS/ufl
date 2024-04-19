"""Typing."""

__all__ = ["Self"]

# TODO: Remove this when Python 3.11 is the minimum version
try:
    from typing import Self
except ImportError:
    from typing import Any as Self

