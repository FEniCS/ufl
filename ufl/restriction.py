"""Restriction operations."""

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
# First added:  2008-06-08
# Last changed: 2011-06-02

from ufl.log import error
from ufl.operatorbase import Operator
from ufl.precedence import parstr
from ufl.common import EmptyDict

#--- Restriction operators ---

class Restricted(Operator):
    __slots__ = ("_f", "_side")

    # TODO: Add __new__ operator here, e.g. restricted(literal) == literal

    def __init__(self, f, side):
        Operator.__init__(self)
        self._f = f
        self._side = side

    def shape(self):
        return self._f.shape()

    def operands(self):
        return (self._f,)

    def free_indices(self):
        return self._f.free_indices()

    def index_dimensions(self):
        return self._f.index_dimensions()

    def evaluate(self, x, mapping, component, index_values):
        return self._f.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "%s('%s')" % (parstr(self._f, self), self._side)

class PositiveRestricted(Restricted):
    __slots__ = ()
    def __init__(self, f):
        Restricted.__init__(self, f, "+")

    def __repr__(self):
        return "PositiveRestricted(%r)" % self._f

class NegativeRestricted(Restricted):
    __slots__ = ()
    def __init__(self, f):
        Restricted.__init__(self, f, "-")

    def __repr__(self):
        return "NegativeRestricted(%r)" % self._f


# TODO: Place in a better file?
class CellAvg(Operator):
    __slots__ = ("_f",)

    # TODO: Add __new__ operator here, e.g. cell_avg(literal) == literal

    def __init__(self, f):
        Operator.__init__(self)
        self._f = f

    def shape(self):
        return self._f.shape()

    def operands(self):
        return (self._f,)

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return EmptyDict

    def evaluate(self, x, mapping, component, index_values):
        "Performs an approximate symbolic evaluation, since we dont have a cell."
        return self._f.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "cell_avg(%s)" % (self._f,)

    def __repr__(self):
        return "CellAvg(%r)" % self._f


# TODO: Place in a better file?
class FacetAvg(Operator):
    __slots__ = ("_f",)

    # TODO: Add __new__ operator here, e.g. facet_avg(literal) == literal

    def __init__(self, f):
        Operator.__init__(self)
        self._f = f

    def shape(self):
        return self._f.shape()

    def operands(self):
        return (self._f,)

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return EmptyDict

    def evaluate(self, x, mapping, component, index_values):
        "Performs an approximate symbolic evaluation, since we dont have a cell."
        return self._f.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "facet_avg(%s)" % (self._f,)

    def __repr__(self):
        return "FacetAvg(%r)" % self._f
