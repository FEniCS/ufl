"""Defines the Variable and Label classes, used to label
expressions as variables for differentiation."""

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from ufl.common import counted_init
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.terminal import UtilityType
from ufl.operatorbase import Operator, WrapperType
from ufl.constantvalue import as_ufl
from ufl.core.ufl_type import ufl_type

@ufl_type()
class Label(UtilityType):
    __slots__ = ("_count",)
    _globalcount = 0
    def __init__(self, count=None):
        UtilityType.__init__(self)
        counted_init(self, count, Label)

    def count(self):
        return self._count

    def __str__(self):
        return "Label(%d)" % self._count

    def __repr__(self):
        return "Label(%d)" % self._count


@ufl_type(is_shaping=True, num_ops=1)
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
    __slots__ = ()

    def __init__(self, expression, label=None):
        # Conversion
        expression = as_ufl(expression)
        if label is None:
            label = Label()

        # Checks
        ufl_assert(isinstance(expression, Expr), "Expecting Expr.")
        ufl_assert(isinstance(label, Label), "Expecting a Label.")

        WrapperType.__init__(self, (expression, label))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape

    def free_indices(self):
        return self.ufl_operands[0].free_indices()

    def index_dimensions(self):
        return self.ufl_operands[0].index_dimensions()

    def domains(self):
        return self.ufl_operands[0].domains()

    def is_cellwise_constant(self):
        return self.ufl_operands[0].is_cellwise_constant()

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return a

    def expression(self):
        return self.ufl_operands[0]

    def label(self):
        return self.ufl_operands[1]

    def __eq__(self, other):
        return (isinstance(other, Variable)
                and self.ufl_operands[1] == other.ufl_operands[1]
                and self.ufl_operands[0] == other.ufl_operands[0])
    __hash__ = Operator.__hash__

    def __str__(self):
        return "var%d(%s)" % (self.ufl_operands[1].count(), self.ufl_operands[0])

    def __repr__(self):
        return "Variable(%r, %r)" % (self.ufl_operands[0], self.ufl_operands[1])

    def __getnewargs__(self):
        return ()
