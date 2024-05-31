"""Restriction operations."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.corealg.map_dag import map_expr_dag
from ufl.precedence import parstr
from ufl.typing import Self, cutoff

# --- Restriction operators ---


@ufl_type(
    is_abstract=True,
    num_ops=1,
    inherit_shape_from_operand=0,
    inherit_indices_from_operand=0,
    is_restriction=True,
)
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
        return self.ufl_operands[0].evaluate(x, mapping, component, index_values)

    def __str__(self):
        """Format as a string."""
        return f"{parstr(self.ufl_operands[0], self)}({self._side})"

    @cutoff
    def apply_restrictions(self, mapped_operands, side) -> Self:
        """Apply restrictions.

        Propagates restrictions in a form towards the terminals.
        """
        # When hitting a restricted quantity, visit child with a separate restriction algorithm
        # Assure that we have only two levels here: inside or outside the Restricted node
        if side is not None:
            raise ValueError("Cannot restrict an expression twice.")
        # Configure a propagator for this side and apply to subtree
        side = self._side
        return map_expr_dag(
            "apply_restrictions",
            (side,),
            self.ufl_operands[0],  # vcache=self.vcaches[side], rcache=self.rcaches[side]
        )


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
