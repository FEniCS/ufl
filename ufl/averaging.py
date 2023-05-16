"""Averaging operations."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.constantvalue import ConstantValue
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type


@ufl_type(inherit_shape_from_operand=0,
          inherit_indices_from_operand=0,
          num_ops=1,
          is_evaluation=True)
class CellAvg(Operator):
    __slots__ = ()

    def __new__(cls, f):
        if isinstance(f, ConstantValue):
            return f
        return super(CellAvg, cls).__new__(cls)

    def __init__(self, f):
        Operator.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape

    def evaluate(self, x, mapping, component, index_values):
        "Performs an approximate symbolic evaluation, since we dont have a cell."
        return self.ufl_operands[0].evaluate(x, mapping, component,
                                             index_values)

    def __str__(self):
        return "cell_avg(%s)" % (self.ufl_operands[0],)


@ufl_type(inherit_shape_from_operand=0,
          inherit_indices_from_operand=0,
          num_ops=1,
          is_evaluation=True)
class FacetAvg(Operator):
    __slots__ = ()

    def __new__(cls, f):
        if isinstance(f, ConstantValue):
            return f
        return super(FacetAvg, cls).__new__(cls)

    def __init__(self, f):
        Operator.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape

    def evaluate(self, x, mapping, component, index_values):
        "Performs an approximate symbolic evaluation, since we dont have a cell."
        return self.ufl_operands[0].evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "facet_avg(%s)" % (self.ufl_operands[0],)
