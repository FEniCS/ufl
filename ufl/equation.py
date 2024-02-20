"""The Equation class, used to express equations like a == L."""
# Copyright (C) 2012-2016 Anders Logg and Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.ufl_type import UFLObject

__all_classes__ = ["Equation"]


class Equation(UFLObject):
    """Equation.

    This class is used to represent equations expressed by the "=="
    operator. Examples include a == L and F == 0 where a, L and F are
    Form objects.
    """

    def __init__(self, lhs, rhs):
        """Create equation lhs == rhs."""
        self.lhs = lhs
        self.rhs = rhs

    def __bool__(self):
        """Evaluate bool(lhs_form == rhs_form).

        This will not trigger when setting 'equation = a == L',
        but when e.g. running 'if equation:'.
        """
        if type(self.lhs) is not type(self.rhs):  # noqa: E721
            return False
        # Try to delegate to equals function
        if hasattr(self.lhs, "equals"):
            return self.lhs.equals(self.rhs)
        elif hasattr(self.rhs, "equals"):
            return self.rhs.equals(self.lhs)
        else:
            raise ValueError("Either lhs or rhs of Equation must implement self.equals(other).")

    __nonzero__ = __bool__

    def __eq__(self, other):
        """Compare two equations by comparing lhs and rhs."""
        return isinstance(other, Equation) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        """Hash."""
        return super().__hash__()

    def _ufl_hash_data_(self):
        """Hash data."""
        return ("Equation", hash(self.lhs), hash(self.rhs))

    def __repr__(self):
        """Representation."""
        return f"Equation({self.lhs!r}, {self.rhs!r})"

    def __str__(self):
        """Format as a string."""
        return f"Equation({self.lhs}, {self.rhs})"
