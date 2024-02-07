"""Representation of the reference value of a function."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.operator import Operator
from ufl.core.terminal import FormArgument
from ufl.restriction import Restricted


class ReferenceValue(Operator):
    """Representation of the reference cell value of a form argument."""

    __slots__ = ()

    def __init__(self, f):
        """Initialise."""
        if not isinstance(f, FormArgument):
            raise ValueError("Can only take reference value of form arguments.")
        Operator.__init__(self, (f,))

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_element().reference_value_shape

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        """Get child from mapping and return the component asked for."""
        raise NotImplementedError()

    def __str__(self):
        """Format as a string."""
        return f"reference_value({self.ufl_operands[0]})"

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        f, = self.ufl_operands
        assert f._is_terminal()
        g = f.apply_restrictions(side)
        if isinstance(g, Restricted):
            side = g.side()
            return self(side)
        else:
            return self

    def get_arity(self):
        """Get the arity."""
        return self.ufl_operands[0].get_arity()
