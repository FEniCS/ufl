# -*- coding: utf-8 -*-
"Basic algebra operations."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008

from ufl.log import error
from ufl.core.ufl_type import ufl_type
from ufl.core.expr import ufl_err_str
from ufl.core.operator import Operator
from ufl.constantvalue import Zero, zero, ScalarValue, IntValue, ComplexValue, as_ufl
from ufl.checks import is_ufl_scalar, is_true_ufl_scalar
from ufl.index_combination_utils import merge_unique_indices
from ufl.sorting import sorted_expr
from ufl.precedence import parstr

# --- Algebraic operators ---


@ufl_type(num_ops=2,
          inherit_shape_from_operand=0, inherit_indices_from_operand=0,
          binop="__add__", rbinop="__radd__")
class Sum(Operator):
    __slots__ = ()

    def __new__(cls, a, b):
        # Make sure everything is an Expr
        a = as_ufl(a)
        b = as_ufl(b)

        # Assert consistent tensor properties
        sh = a.ufl_shape
        fi = a.ufl_free_indices
        fid = a.ufl_index_dimensions
        if b.ufl_shape != sh:
            error("Can't add expressions with different shapes.")
        if b.ufl_free_indices != fi:
            error("Can't add expressions with different free indices.")
        if b.ufl_index_dimensions != fid:
            error("Can't add expressions with different index dimensions.")

        # Skip adding zero
        if isinstance(a, Zero):
            return b
        elif isinstance(b, Zero):
            return a

        # Handle scalars specially and sort operands
        sa = isinstance(a, ScalarValue)
        sb = isinstance(b, ScalarValue)
        if sa and sb:
            # Apply constant propagation
            return as_ufl(a._value + b._value)
        elif sa:
            # Place scalar first
            # operands = (a, b)
            pass  # a, b = a, b
        elif sb:
            # Place scalar first
            # operands = (b, a)
            a, b = b, a
        # elif a == b:
        #    # Replace a+b with 2*foo
        #    return 2*a
        else:
            # Otherwise sort operands in a canonical order
            # operands = (b, a)
            a, b = sorted_expr((a, b))

        # construct and initialize a new Sum object
        self = Operator.__new__(cls)
        self._init(a, b)
        return self

    def _init(self, a, b):
        self.ufl_operands = (a, b)

    def __init__(self, a, b):
        Operator.__init__(self)

    def evaluate(self, x, mapping, component, index_values):
        return sum(o.evaluate(x, mapping, component,
                              index_values) for o in self.ufl_operands)

    def __str__(self):
        ops = [parstr(o, self) for o in self.ufl_operands]
        if False:
            # Implementation with line splitting:
            limit = 70
            delimop = " + \\\n    + "
            op = " + "
            s = ops[0]
            n = len(s)
            for o in ops[1:]:
                m = len(o)
                if n + m > limit:
                    s += delimop
                    n = m
                else:
                    s += op
                    n += m
                s += o
            return s
        # Implementation with no line splitting:
        return "%s" % " + ".join(ops)


@ufl_type(num_ops=2,
          binop="__mul__", rbinop="__rmul__")
class Product(Operator):
    """The product of two or more UFL objects."""
    __slots__ = ("ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, a, b):
        # Conversion
        a = as_ufl(a)
        b = as_ufl(b)

        # Type checking
        # Make sure everything is scalar
        if a.ufl_shape or b.ufl_shape:
            error("Product can only represent products of scalars, "
                  "got\n\t%s\nand\n\t%s" % (ufl_err_str(a), ufl_err_str(b)))

        # Simplification
        if isinstance(a, Zero) or isinstance(b, Zero):
            # Got any zeros? Return zero.
            fi, fid = merge_unique_indices(a.ufl_free_indices,
                                           a.ufl_index_dimensions,
                                           b.ufl_free_indices,
                                           b.ufl_index_dimensions)
            return Zero((), fi, fid)
        sa = isinstance(a, ScalarValue)
        sb = isinstance(b, ScalarValue)
        if sa and sb:  # const * const = const
            # FIXME: Handle free indices like with zero? I think
            # IntValue may be index annotated now?
            return as_ufl(a._value * b._value)
        elif sa:  # 1 * b = b
            if a._value == 1:
                return b
            # a, b = a, b
        elif sb:  # a * 1 = a
            if b._value == 1:
                return a
            a, b = b, a
        # elif a == b: # a * a = a**2 # TODO: Why? Maybe just remove this?
        #    if not a.ufl_free_indices:
        #        return a**2
        else:  # a * b = b * a
            # Sort operands in a semi-canonical order
            # (NB! This is fragile! Small changes here can have large effects.)
            a, b = sorted_expr((a, b))

        # Construction
        self = Operator.__new__(cls)
        self._init(a, b)
        return self

    def _init(self, a, b):
        "Constructor, called by __new__ with already checked arguments."
        self.ufl_operands = (a, b)

        # Extract indices
        fi, fid = merge_unique_indices(a.ufl_free_indices,
                                       a.ufl_index_dimensions,
                                       b.ufl_free_indices,
                                       b.ufl_index_dimensions)
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    def __init__(self, a, b):
        Operator.__init__(self)

    ufl_shape = ()

    def evaluate(self, x, mapping, component, index_values):
        ops = self.ufl_operands
        sh = self.ufl_shape
        if sh:
            if sh != ops[-1].ufl_shape:
                error("Expecting nonscalar product operand to be the last by convention.")
            tmp = ops[-1].evaluate(x, mapping, component, index_values)
            ops = ops[:-1]
        else:
            tmp = 1
        for o in ops:
            tmp *= o.evaluate(x, mapping, (), index_values)
        return tmp

    def __str__(self):
        a, b = self.ufl_operands
        return " * ".join((parstr(a, self), parstr(b, self)))


@ufl_type(num_ops=2,
          inherit_indices_from_operand=0,
          binop="__div__", rbinop="__rdiv__")
class Division(Operator):
    __slots__ = ()

    def __new__(cls, a, b):
        # Conversion
        a = as_ufl(a)
        b = as_ufl(b)

        # Type checking
        # TODO: Enabled workaround for nonscalar division in __div__,
        # so maybe we can keep this assertion. Some algorithms may
        # need updating.
        if not is_ufl_scalar(a):
            error("Expecting scalar nominator in Division.")
        if not is_true_ufl_scalar(b):
            error("Division by non-scalar is undefined.")
        if isinstance(b, Zero):
            error("Division by zero!")

        # Simplification
        # Simplification a/b -> a
        if isinstance(a, Zero) or (isinstance(b, ScalarValue) and b._value == 1):
            return a
        # Simplification "literal a / literal b" -> "literal value of
        # a/b". Avoiding integer division by casting to float
        if isinstance(a, ScalarValue) and isinstance(b, ScalarValue):
            try:
                return as_ufl(float(a._value) / float(b._value))
            except TypeError:
                return as_ufl(complex(a._value) / complex(b._value))
        # Simplification "a / a" -> "1"
        # if not a.ufl_free_indices and not a.ufl_shape and a == b:
        #    return as_ufl(1)

        # Construction
        self = Operator.__new__(cls)
        self._init(a, b)
        return self

    def _init(self, a, b):
        self.ufl_operands = (a, b)

    def __init__(self, a, b):
        Operator.__init__(self)

    ufl_shape = ()  # self.ufl_operands[0].ufl_shape

    def evaluate(self, x, mapping, component, index_values):
        a, b = self.ufl_operands
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        # Avoiding integer division by casting to float
        try:
            e = float(a) / float(b)
        except TypeError:
            e = complex(a) / complex(b)
        return e

    def __str__(self):
        return "%s / %s" % (parstr(self.ufl_operands[0], self),
                            parstr(self.ufl_operands[1], self))


@ufl_type(num_ops=2,
          inherit_indices_from_operand=0,
          binop="__pow__", rbinop="__rpow__")
class Power(Operator):
    __slots__ = ()

    def __new__(cls, a, b):
        # Conversion
        a = as_ufl(a)
        b = as_ufl(b)

        # Type checking
        if not is_true_ufl_scalar(a):
            error("Cannot take the power of a non-scalar expression %s." % ufl_err_str(a))
        if not is_true_ufl_scalar(b):
            error("Cannot raise an expression to a non-scalar power %s." % ufl_err_str(b))

        # Simplification
        if isinstance(a, ScalarValue) and isinstance(b, ScalarValue):
            return as_ufl(a._value ** b._value)
        if isinstance(b, Zero):
            return IntValue(1)
        if isinstance(a, Zero) and isinstance(b, ScalarValue):
            if isinstance(b, ComplexValue):
                error("Cannot raise zero to a complex power.")
            bf = float(b)
            if bf < 0:
                error("Division by zero, cannot raise 0 to a negative power.")
            else:
                return zero()
        if isinstance(b, ScalarValue) and b._value == 1:
            return a

        # Construction
        self = Operator.__new__(cls)
        self._init(a, b)
        return self

    def _init(self, a, b):
        self.ufl_operands = (a, b)

    def __init__(self, a, b):
        Operator.__init__(self)

    ufl_shape = ()

    def evaluate(self, x, mapping, component, index_values):
        a, b = self.ufl_operands
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        return a**b

    def __str__(self):
        a, b = self.ufl_operands
        return "%s ** %s" % (parstr(a, self), parstr(b, self))


@ufl_type(num_ops=1,
          inherit_shape_from_operand=0, inherit_indices_from_operand=0,
          unop="__abs__")
class Abs(Operator):
    __slots__ = ()

    def __new__(cls, a):
        a = as_ufl(a)

        # Simplification
        if isinstance(a, (Zero, Abs)):
            return a
        if isinstance(a, Conj):
            return Abs(a.ufl_operands[0])
        if isinstance(a, ScalarValue):
            return as_ufl(abs(a._value))

        return Operator.__new__(cls)

    def __init__(self, a):
        Operator.__init__(self, (a,))

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return abs(a)

    def __str__(self):
        a, = self.ufl_operands
        return "|%s|" % (parstr(a, self),)


@ufl_type(num_ops=1,
          inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Conj(Operator):
    __slots__ = ()

    def __new__(cls, a):
        a = as_ufl(a)

        # Simplification
        if isinstance(a, (Abs, Real, Imag, Zero)):
            return a
        if isinstance(a, Conj):
            return a.ufl_operands[0]
        if isinstance(a, ScalarValue):
            return as_ufl(a._value.conjugate())

        return Operator.__new__(cls)

    def __init__(self, a):
        Operator.__init__(self, (a,))

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return a.conjugate()

    def __str__(self):
        a, = self.ufl_operands
        return "conj(%s)" % (parstr(a, self),)


@ufl_type(num_ops=1,
          inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Real(Operator):
    __slots__ = ()

    def __new__(cls, a):
        a = as_ufl(a)

        # Simplification
        if isinstance(a, Conj):
            a = a.ufl_operands[0]
        if isinstance(a, Zero):
            return a
        if isinstance(a, ScalarValue):
            return as_ufl(a.real())
        if isinstance(a, Real):
            a = a.ufl_operands[0]

        return Operator.__new__(cls)

    def __init__(self, a):
        Operator.__init__(self, (a,))

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return a.real

    def __str__(self):
        a, = self.ufl_operands
        return "Re[%s]" % (parstr(a, self),)


@ufl_type(num_ops=1,
          inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Imag(Operator):
    __slots__ = ()

    def __new__(cls, a):
        a = as_ufl(a)

        # Simplification
        if isinstance(a, Zero):
            return a
        if isinstance(a, (Real, Imag, Abs)):
            return Zero(a.ufl_shape, a.ufl_free_indices, a.ufl_index_dimensions)
        if isinstance(a, ScalarValue):
            return as_ufl(a.imag())

        return Operator.__new__(cls)

    def __init__(self, a):
        Operator.__init__(self, (a,))

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return a.imag

    def __str__(self):
        a, = self.ufl_operands
        return "Im[%s]" % (parstr(a, self),)
