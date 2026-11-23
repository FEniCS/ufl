"""Averaging operations."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.constantvalue import ConstantValue
from ufl.core.operator import Operator, ShapeAndIndicesFromFirstOperand
from ufl.core.ufl_type import ufl_type


@ufl_type
class CellAvg(ShapeAndIndicesFromFirstOperand, Operator):
    """Cell average."""

    __slots__ = ()

    def __new__(cls, f):
        """Create a new CellAvg."""
        if isinstance(f, ConstantValue):
            return f
        return super().__new__(cls)

    def __init__(self, f):
        """Initialise."""
        Operator.__init__(self, (f,))

    def evaluate(self, x, mapping, component, index_values):
        """Performs an approximate symbolic evaluation, since we don't have a cell."""
        return self.ufl_operands[0].evaluate(x, mapping, component, index_values)

    def __str__(self):
        """Format as a string."""
        return f"cell_avg({self.ufl_operands[0]})"


@ufl_type
class FacetAvg(ShapeAndIndicesFromFirstOperand, Operator):
    """Facet average."""

    __slots__ = ()

    def __new__(cls, f):
        """Create a new FacetAvg."""
        if isinstance(f, ConstantValue):
            return f
        return super().__new__(cls)

    def __init__(self, f):
        """Initialise."""
        Operator.__init__(self, (f,))

    def evaluate(self, x, mapping, component, index_values):
        """Performs an approximate symbolic evaluation, since we dont have a cell."""
        return self.ufl_operands[0].evaluate(x, mapping, component, index_values)

    def __str__(self):
        """Format as a string."""
        return f"facet_avg({self.ufl_operands[0]})"
