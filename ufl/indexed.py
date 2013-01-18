"""This module defines the Indexed class."""

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

from itertools import izip
from ufl.log import error
from ufl.expr import Expr
from ufl.operatorbase import WrapperType
from ufl.indexing import Index, FixedIndex, as_multi_index
from ufl.indexutils import unique_indices
from ufl.precedence import parstr
from ufl.common import EmptyDict

#--- Indexed expression ---

class Indexed(WrapperType):
    __slots__ = ("_expression", "_indices",
                 "_free_indices", "_index_dimensions",)
    def __init__(self, expression, indices):
        WrapperType.__init__(self)
        if not isinstance(expression, Expr):
            error("Expecting Expr instance, not %s." % repr(expression))
        self._expression = expression

        shape = expression.shape()
        if len(shape) != len(indices):
            error("Invalid number of indices (%d) for tensor "\
                "expression of rank %d:\n\t%r\n"\
                % (len(indices), expression.rank(), expression))

        self._indices = as_multi_index(indices, shape)

        for si, di in izip(shape, self._indices):
            if isinstance(di, FixedIndex) and int(di) >= int(si):
                error("Fixed index out of range!")

        idims = dict((i, s) for (i, s) in izip(self._indices._indices, shape)
                     if isinstance(i, Index))
        idims.update(expression.index_dimensions())
        fi = unique_indices(expression.free_indices() + self._indices._indices)

        self._free_indices = fi
        self._index_dimensions = idims or EmptyDict

    def operands(self):
        return (self._expression, self._indices)

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def shape(self):
        return ()

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return self._expression.is_cellwise_constant()

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        A, ii = self.operands()
        component = ii.evaluate(x, mapping, None, index_values)
        if derivatives:
            return A.evaluate(x, mapping, component, index_values, derivatives)
        else:
            return A.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "%s[%s]" % (parstr(self._expression, self), self._indices)

    def __repr__(self):
        return "Indexed(%r, %r)" % (self._expression, self._indices)

    def __getitem__(self, key):
        error("Attempting to index with %r, but object is already indexed: %r" % (key, self))

    def __getnewargs__(self):
        return ()
