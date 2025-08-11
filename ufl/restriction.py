"""Restriction operations."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.constantvalue import ConstantValue
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.precedence import parstr

# --- Restriction operators ---


@ufl_type(
    inherit_indices_from_operand=0,
)
class Restricted(Operator):
    """Restriction."""

    _ufl_is_restriction_ = True
    __slots__ = ()
    _side: str

    def __new__(cls, expression):
        """Create a new Restricted."""
        if isinstance(expression, ConstantValue):
            return expression
        else:
            return Operator.__new__(cls)

    def __init__(self, f):
        """Initialise."""
        Operator.__init__(self, (f,))

    def side(self):
        """Get the side."""
        return self._side

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        return self.ufl_operands[0].evaluate(x, mapping, component, index_values)

    def __str__(self):
        """Format as a string."""
        return f"{parstr(self.ufl_operands[0], self)}({self._side})"

    @property
    def ufl_shape(self):
        """Return shape."""
        return self.ufl_operands[0].ufl_shape


@ufl_type()
class PositiveRestricted(Restricted):
    """Positive restriction."""

    _ufl_is_terminal_modifier_ = True
    __slots__ = ()
    _side = "+"


@ufl_type()
class NegativeRestricted(Restricted):
    """Negative restriction."""

    _ufl_is_terminal_modifier_ = True
    __slots__ = ()
    _side = "-"
