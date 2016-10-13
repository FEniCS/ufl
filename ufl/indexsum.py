# -*- coding: utf-8 -*-
"""This module defines the IndexSum class."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.

from six.moves import xrange as range

from ufl.log import error
from ufl.utils.py23 import as_native_strings
from ufl.core.ufl_type import ufl_type
from ufl.core.expr import Expr, ufl_err_str
from ufl.core.operator import Operator
from ufl.core.multiindex import MultiIndex
from ufl.precedence import parstr
from ufl.constantvalue import Zero


# --- Sum over an index ---

@ufl_type(num_ops=2)
class IndexSum(Operator):
    __slots__ = as_native_strings((
        "_dimension",
        "ufl_free_indices",
        "ufl_index_dimensions",
    ))

    def __new__(cls, summand, index):
        # Error checks
        if not isinstance(summand, Expr):
            error("Expecting Expr instance, got %s" % ufl_err_str(summand))
        if not isinstance(index, MultiIndex):
            error("Expecting MultiIndex instance, got %s" % ufl_err_str(index))
        if len(index) != 1:
            error("Expecting a single Index onlym got %d." % len(index))

        # Simplification to zero
        if isinstance(summand, Zero):
            sh = summand.ufl_shape
            j, = index
            fi = summand.ufl_free_indices
            fid = summand.ufl_index_dimensions
            pos = fi.index(j.count())
            fi = fi[:pos] + fi[pos+1:]
            fid = fid[:pos] + fid[pos+1:]
            return Zero(sh, fi, fid)

        return Operator.__new__(cls)

    def __init__(self, summand, index):
        j, = index
        fi = summand.ufl_free_indices
        fid = summand.ufl_index_dimensions
        pos = fi.index(j.count())
        self._dimension = fid[pos]
        self.ufl_free_indices = fi[:pos] + fi[pos+1:]
        self.ufl_index_dimensions = fid[:pos] + fid[pos+1:]
        Operator.__init__(self, (summand, index))

    def index(self):
        return self.ufl_operands[1][0]

    def dimension(self):
        return self._dimension

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape

    def evaluate(self, x, mapping, component, index_values):
        i, = self.ufl_operands[1]
        tmp = 0
        for k in range(self._dimension):
            index_values.push(i, k)
            tmp += self.ufl_operands[0].evaluate(x, mapping, component,
                                                 index_values)
            index_values.pop()
        return tmp

    def __str__(self):
        return "sum_{%s} %s " % (str(self.ufl_operands[1]),
                                 parstr(self.ufl_operands[0], self))
