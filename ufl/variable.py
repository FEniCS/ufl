"""Defines the Variable and Label classes, used to label
expressions as variables for differentiation."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
# First added:  2008-05-20
# Last changed: 2011-06-02

from ufl.common import Counted
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.expr import Expr, WrapperType
from ufl.terminal import UtilityType
from ufl.constantvalue import as_ufl

class Label(UtilityType, Counted):
    _globalcount = 0
    __slots__ = ("_repr", "_hash")
    def __init__(self, count=None):
        Counted.__init__(self, count)
        self._repr = "Label(%d)" % self._count
        self._hash = hash(self._repr)

    def __str__(self):
        return "Label(%d)" % self._count

    def __repr__(self):
        return self._repr

    def __hash__(self):
        return self._hash

class Variable(WrapperType):
    """A Variable is a representative for another expression.

    It will be used by the end-user mainly for defining
    a quantity to differentiate w.r.t. using diff.
    Example::

      e = <...>
      e = variable(e)
      f = exp(e**2)
      df = diff(f, e)
    """
    __slots__ = ("_expression", "_label", "_repr")
    def __init__(self, expression, label=None):
        WrapperType.__init__(self)
        expression = as_ufl(expression)
        ufl_assert(isinstance(expression, Expr), "Expecting Expr.")
        self._expression = expression

        if label is None:
            label = Label()
        ufl_assert(isinstance(label, Label), "Expecting a Label.")
        self._label = label
        self._repr = "Variable(%r, %r)" % (self._expression, self._label)

    def operands(self):
        return (self._expression, self._label)

    def free_indices(self):
        return self._expression.free_indices()

    def index_dimensions(self):
        return self._expression.index_dimensions()

    def shape(self):
        return self._expression.shape()

    def cell(self):
        return self._expression.cell()

    def is_cellwise_constant(self):
        return self._expression.is_cellwise_constant()

    def evaluate(self, x, mapping, component, index_values):
        a = self._expression.evaluate(x, mapping, component, index_values)
        return a

    def expression(self):
        return self._expression

    def label(self):
        return self._label

    def __eq__(self, other):
        return isinstance(other, Variable) and self._label == other._label and self._expression == other._expression

    def __str__(self):
        #return "Variable(%s, %s)" % (self._expression, self._label)
        return "var%d(%s)" % (self._label.count(), self._expression)

    def __repr__(self):
        return self._repr
