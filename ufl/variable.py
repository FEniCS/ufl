# -*- coding: utf-8 -*-
"""Defines the Variable and Label classes, used to label
expressions as variables for differentiation."""

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

from ufl.utils.counted import counted_init
from ufl.log import error
from ufl.core.expr import Expr
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import Terminal
from ufl.core.operator import Operator
from ufl.constantvalue import as_ufl


@ufl_type()
class Label(Terminal):
    __slots__ = ("_count",)

    _globalcount = 0

    def __init__(self, count=None):
        Terminal.__init__(self)
        counted_init(self, count, Label)

    def count(self):
        return self._count

    def __str__(self):
        return "Label(%d)" % self._count

    def __repr__(self):
        r = "Label(%d)" % self._count
        return r

    @property
    def ufl_shape(self):
        error("Label has no shape (it is not a tensor expression).")

    @property
    def ufl_free_indices(self):
        error("Label has no free indices (it is not a tensor expression).")

    @property
    def ufl_index_dimensions(self):
        error("Label has no free indices (it is not a tensor expression).")

    def is_cellwise_constant(self):
        return True

    def ufl_domains(self):
        "Return tuple of domains related to this terminal object."
        return ()


@ufl_type(is_shaping=True, is_index_free=True, num_ops=1, inherit_shape_from_operand=0)
class Variable(Operator):
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
        if not isinstance(expression, Expr):
            error("Expecting Expr.")
        if not isinstance(label, Label):
            error("Expecting a Label.")
        if expression.ufl_free_indices:
            error("Variable cannot wrap an expression with free indices.")

        Operator.__init__(self, (expression, label))

    def ufl_domains(self):
        return self.ufl_operands[0].ufl_domains()

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return a

    def expression(self):
        return self.ufl_operands[0]

    def label(self):
        return self.ufl_operands[1]

    def __eq__(self, other):
        return (isinstance(other, Variable) and
                self.ufl_operands[1] == other.ufl_operands[1] and
                self.ufl_operands[0] == other.ufl_operands[0])

    def __str__(self):
        return "var%d(%s)" % (self.ufl_operands[1].count(),
                              self.ufl_operands[0])
