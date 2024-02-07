"""Restriction operations."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.operator import Operator
from ufl.precedence import parstr
from abc import abstractproperty

default_restriction = "+"


def require_restriction(o):
    """Raise an error."""
    raise ValueError(f"Discontinuous type {o._ufl_class_.__name__} must be restricted.")


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

    @property
    def ufl_shape(self):
        """Shape."""
        return self.ufl_operands[0].ufl_shape

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self

    def apply_restrictions(self, side=None):
        """Apply default restrictions."""
        if side is not None:
            raise ValueError("Cannot restrict an expression twice.")
        return self.__class__(self.ufl_operands[0].apply_restrictions(self._side))

    @property
    def ufl_free_indices(self):
        """A tuple of free index counts."""
        return self.ufl_operands[0].ufl_free_indices

    @property
    def ufl_index_dimensions(self):
        """A tuple providing the int dimension for each free index."""
        return self.ufl_operands[0].ufl_index_dimensions

    @abstractproperty
    def _side(self):
        """The side on which the restriction is taken."""

    def get_arity(self):
        """Get the arity."""
        return self.ufl_operands[0].get_arity()


class PositiveRestricted(Restricted):
    """Positive restriction."""

    __slots__ = ()

    @property
    def _side(self):
        """The side on which the restriction is taken."""
        return "+"

    def __repr__(self):
        """Format as a string."""
        return f"PositiveRestricted({self.ufl_operands[0]!r})"


class NegativeRestricted(Restricted):
    """Negative restriction."""

    __slots__ = ()

    @property
    def _side(self):
        """The side on which the restriction is taken."""
        return "-"

    def __repr__(self):
        """Format as a string."""
        return f"NegativeRestricted({self.ufl_operands[0]!r})"
