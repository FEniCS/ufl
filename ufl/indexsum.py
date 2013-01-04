"""This module defines the IndexSum class."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
#
# First added:  2009-01-28
# Last changed: 2011-06-17

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.operatorbase import AlgebraOperator
from ufl.indexing import Index, MultiIndex, as_multi_index
from ufl.precedence import parstr
from ufl.common import EmptyDict

#--- Sum over an index ---

class IndexSum(AlgebraOperator):
    __slots__ = ("_summand", "_index", "_dimension", "_free_indices", "_index_dimensions")

    def __new__(cls, summand, index):
        if not isinstance(summand, Expr):
            error("Expecting Expr instance, not %s." % repr(summand))

        from ufl.constantvalue import Zero
        if isinstance(summand, Zero):
            sh = summand.shape()

            if isinstance(index, Index):
                j = index
            elif isinstance(index, MultiIndex):
                if len(index) != 1:
                    error("Expecting a single Index only.")
                j, = index

            fi = tuple(i for i in summand.free_indices() if not i == j)
            idims = dict(summand.index_dimensions())
            del idims[j]
            return Zero(sh, fi, idims)
        return AlgebraOperator.__new__(cls)

    def __init__(self, summand, index):
        AlgebraOperator.__init__(self)

        if isinstance(index, Index):
            j = index
        elif isinstance(index, MultiIndex):
            if len(index) != 1:
                error("Expecting a single Index only.")
            j, = index

        self._summand = summand
        self._index_dimensions = dict(summand.index_dimensions())
        self._free_indices = tuple(i for i in summand.free_indices() if not i == j)

        d = self._index_dimensions[j]
        self._index = as_multi_index(index, (d,))
        ufl_assert(isinstance(self._index, MultiIndex), "Error in initialization of index sum.")
        self._dimension = d
        del self._index_dimensions[j]
        if not self._index_dimensions:
            self._index_dimensions = EmptyDict

    def index(self):
        return self._index[0]

    def dimension(self):
        return self._dimension

    def operands(self):
        return (self._summand, self._index)

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def shape(self):
        return self._summand.shape()

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return self._summand.is_cellwise_constant()

    def evaluate(self, x, mapping, component, index_values):
        i, = self._index
        tmp = 0
        for k in range(self._dimension):
            index_values.push(i, k)
            tmp += self._summand.evaluate(x, mapping, component, index_values)
            index_values.pop()
        return tmp

    def __str__(self):
        return "sum_{%s} %s " % (str(self._index), parstr(self._summand, self))

    def __repr__(self):
        return "IndexSum(%r, %r)" % (self._summand, self._index)

