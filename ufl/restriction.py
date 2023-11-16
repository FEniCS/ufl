"""Restriction operations."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.precedence import parstr

# --- Restriction operators ---


@ufl_type(is_abstract=True,
          num_ops=1,
          inherit_shape_from_operand=0,
          inherit_indices_from_operand=0,
          is_restriction=True)
class Restricted(Operator):
    """Restriction."""

    __slots__ = ()

    # TODO: Add __new__ operator here, e.g. restricted(literal) == literal

    def __init__(self, f):
        """Initialise."""
        Operator.__init__(self, (f,))

    def side(self):
        """Get the side."""
        return self._side

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        return self.ufl_operands[0].evaluate(x, mapping, component,
                                             index_values)

    def __str__(self):
        """Format as a string."""
        return f"{parstr(self.ufl_operands[0], self)}({self._side})"


@ufl_type(is_terminal_modifier=True)
class PositiveRestricted(Restricted):
    """Positive restriction."""

    __slots__ = ()
    _side = "+"


@ufl_type(is_terminal_modifier=True)
class NegativeRestricted(Restricted):
    """Negative restriction."""

    __slots__ = ()
    _side = "-"
