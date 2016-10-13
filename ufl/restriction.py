# -*- coding: utf-8 -*-
"""Restriction operations."""

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

from ufl.core.operator import Operator
from ufl.precedence import parstr
from ufl.core.ufl_type import ufl_type


# --- Restriction operators ---

@ufl_type(is_abstract=True,
          num_ops=1,
          inherit_shape_from_operand=0,
          inherit_indices_from_operand=0,
          is_restriction=True)
class Restricted(Operator):
    __slots__ = ()

    # TODO: Add __new__ operator here, e.g. restricted(literal) == literal

    def __init__(self, f):
        Operator.__init__(self, (f,))

    def side(self):
        return self._side

    def evaluate(self, x, mapping, component, index_values):
        return self.ufl_operands[0].evaluate(x, mapping, component,
                                             index_values)

    def __str__(self):
        return "%s('%s')" % (parstr(self.ufl_operands[0], self), self._side)


@ufl_type(is_terminal_modifier=True)
class PositiveRestricted(Restricted):
    __slots__ = ()
    _side = "+"


@ufl_type(is_terminal_modifier=True)
class NegativeRestricted(Restricted):
    __slots__ = ()
    _side = "-"


# TODO: Place in a better file?
@ufl_type(is_index_free=True,
          num_ops=1,
          is_terminal_modifier=True,
          is_evaluation=True)
class CellAvg(Operator):
    __slots__ = ()

    # TODO: Add __new__ operator here, e.g. cell_avg(literal) == literal

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


# TODO: Place in a better file?
@ufl_type(is_index_free=True,
          num_ops=1,
          is_terminal_modifier=True,
          is_evaluation=True)
class FacetAvg(Operator):
    __slots__ = ()

    # TODO: Add __new__ operator here, e.g. facet_avg(literal) == literal

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
