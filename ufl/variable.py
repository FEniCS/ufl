# -*- coding: utf-8 -*-
"""Defines the Variable and Label classes, used to label
expressions as variables for differentiation."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.constantvalue import as_ufl
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.core.terminal import Terminal
from ufl.core.ufl_type import ufl_type
from ufl.utils.counted import counted_init


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
        raise ValueError("Label has no shape (it is not a tensor expression).")

    @property
    def ufl_free_indices(self):
        raise ValueError("Label has no free indices (it is not a tensor expression).")

    @property
    def ufl_index_dimensions(self):
        raise ValueError("Label has no free indices (it is not a tensor expression).")

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
            raise ValueError("Expecting Expr.")
        if not isinstance(label, Label):
            raise ValueError("Expecting a Label.")
        if expression.ufl_free_indices:
            raise ValueError("Variable cannot wrap an expression with free indices.")

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
