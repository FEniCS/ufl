"""Define the Variable and Label classes.

These are used to label expressions as variables for differentiation.
"""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.utils.counted import Counted
from ufl.core.expr import Expr
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import Terminal
from ufl.core.operator import Operator
from ufl.constantvalue import as_ufl


@ufl_type()
class Label(Terminal, Counted):
    """Label."""

    __slots__ = ("_count", "_counted_class")

    def __init__(self, count=None):
        """Initialise."""
        Terminal.__init__(self)
        Counted.__init__(self, count, Label)

    def __str__(self):
        """Format as a string."""
        return "Label(%d)" % self._count

    def __repr__(self):
        """Representation."""
        r = "Label(%d)" % self._count
        return r

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        raise ValueError("Label has no shape (it is not a tensor expression).")

    @property
    def ufl_free_indices(self):
        """Get the UFL free indices."""
        raise ValueError("Label has no free indices (it is not a tensor expression).")

    @property
    def ufl_index_dimensions(self):
        """Get the UFL index dimensions."""
        raise ValueError("Label has no free indices (it is not a tensor expression).")

    def is_cellwise_constant(self):
        """Return true if the object is constant on each cell."""
        return True

    def ufl_domains(self):
        """Return tuple of domains related to this terminal object."""
        return ()

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        if self not in renumbering:
            return ("Label", self._count)
        return ("Label", renumbering[self])


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
        """Initalise."""
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
        """Get the UFL domains."""
        return self.ufl_operands[0].ufl_domains()

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return a

    def expression(self):
        """Get expression."""
        return self.ufl_operands[0]

    def label(self):
        """Get label."""
        return self.ufl_operands[1]

    def __eq__(self, other):
        """Check equality."""
        return (isinstance(other, Variable) and self.ufl_operands[1] == other.ufl_operands[1] and  # noqa: W504
                self.ufl_operands[0] == other.ufl_operands[0])

    def __str__(self):
        """Format as a string."""
        return "var%d(%s)" % (self.ufl_operands[1].count(),
                              self.ufl_operands[0])
