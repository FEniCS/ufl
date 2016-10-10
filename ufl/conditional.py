# -*- coding: utf-8 -*-
"""This module defines classes for conditional expressions."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.core.operator import Operator
from ufl.constantvalue import as_ufl
from ufl.precedence import parstr
from ufl.exprequals import expr_equals
from ufl.checks import is_true_ufl_scalar
from ufl.core.ufl_type import ufl_type

# --- Condition classes ---

# TODO: Would be nice with some kind of type system to show that this
# is a boolean type not a float type


@ufl_type(is_abstract=True, is_scalar=True)
class Condition(Operator):
    __slots__ = ()

    def __init__(self, operands):
        Operator.__init__(self, operands)

    def __bool__(self):
        # Showing explicit error here to protect against misuse
        error("UFL conditions cannot be evaluated as bool in a Python context.")
    __nonzero__ = __bool__


@ufl_type(is_abstract=True, num_ops=2)
class BinaryCondition(Condition):
    __slots__ = ('_name',)

    def __init__(self, name, left, right):
        left = as_ufl(left)
        right = as_ufl(right)

        Condition.__init__(self, (left, right))

        self._name = name

        if name in ('!=', '=='):
            # Since equals and not-equals are used for comparing
            # representations, we have to allow any shape here. The
            # scalar properties must be checked when used in
            # conditional instead!
            pass
        elif name in ('&&', '||'):
            # Binary operators acting on boolean expressions allow
            # only conditions
            ufl_assert(isinstance(left, Condition),
                       "Expecting a Condition, not a %s." % left._ufl_class_)
            ufl_assert(isinstance(right, Condition),
                       "Expecting a Condition, not a %s." % right._ufl_class_)
        else:
            # Binary operators acting on non-boolean expressions allow
            # only scalars
            ufl_assert(left.ufl_shape == () and right.ufl_shape == (),
                       "Expecting scalar arguments.")
            ufl_assert(left.ufl_free_indices == () and
                       right.ufl_free_indices == (),
                       "Expecting scalar arguments.")

    def __str__(self):
        return "%s %s %s" % (parstr(self.ufl_operands[0], self),
                             self._name, parstr(self.ufl_operands[1], self))


# Not associating with __eq__, the concept of equality with == is
# reserved for object equivalence for use in set and dict.
@ufl_type()
class EQ(BinaryCondition):
    __slots__ = ()

    def __init__(self, left, right):
        BinaryCondition.__init__(self, "==", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a == b)

    def __bool__(self):
        return expr_equals(self.ufl_operands[0], self.ufl_operands[1])
    __nonzero__ = __bool__


# Not associating with __ne__, the concept of equality with == is
# reserved for object equivalence for use in set and dict.
@ufl_type()
class NE(BinaryCondition):
    __slots__ = ()

    def __init__(self, left, right):
        BinaryCondition.__init__(self, "!=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a != b)

    def __bool__(self):
        return not expr_equals(self.ufl_operands[0], self.ufl_operands[1])
    __nonzero__ = __bool__


@ufl_type(binop="__le__")
class LE(BinaryCondition):
    __slots__ = ()

    def __init__(self, left, right):
        BinaryCondition.__init__(self, "<=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a <= b)


@ufl_type(binop="__ge__")
class GE(BinaryCondition):
    __slots__ = ()

    def __init__(self, left, right):
        BinaryCondition.__init__(self, ">=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a >= b)


@ufl_type(binop="__lt__")
class LT(BinaryCondition):
    __slots__ = ()

    def __init__(self, left, right):
        BinaryCondition.__init__(self, "<", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a < b)


@ufl_type(binop="__gt__")
class GT(BinaryCondition):
    __slots__ = ()

    def __init__(self, left, right):
        BinaryCondition.__init__(self, ">", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a > b)


@ufl_type()
class AndCondition(BinaryCondition):
    __slots__ = ()

    def __init__(self, left, right):
        BinaryCondition.__init__(self, "&&", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a and b)


@ufl_type()
class OrCondition(BinaryCondition):
    __slots__ = ()

    def __init__(self, left, right):
        BinaryCondition.__init__(self, "||", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a or b)


@ufl_type(num_ops=1)
class NotCondition(Condition):
    __slots__ = ()

    def __init__(self, condition):
        Condition.__init__(self, (condition,))
        ufl_assert(isinstance(condition, Condition), "Expecting a condition.")

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return bool(not a)

    def __str__(self):
        return "!(%s)" % (str(self.ufl_operands[0]),)


# --- Conditional expression (condition ? true_value : false_value) ---

@ufl_type(num_ops=3, inherit_shape_from_operand=1,
          inherit_indices_from_operand=1)
class Conditional(Operator):
    __slots__ = ()

    def __init__(self, condition, true_value, false_value):
        ufl_assert(isinstance(condition, Condition),
                   "Expectiong condition as first argument.")
        true_value = as_ufl(true_value)
        false_value = as_ufl(false_value)
        tsh = true_value.ufl_shape
        fsh = false_value.ufl_shape
        ufl_assert(tsh == fsh, "Shape mismatch between conditional branches.")
        tfi = true_value.ufl_free_indices
        ffi = false_value.ufl_free_indices
        ufl_assert(tfi == ffi,
                   "Free index mismatch between conditional branches.")
        if isinstance(condition, (EQ, NE)):
            ufl_assert(condition.ufl_operands[0].ufl_shape == () and
                       condition.ufl_operands[0].ufl_free_indices == () and
                       condition.ufl_operands[1].ufl_shape == () and
                       condition.ufl_operands[1].ufl_free_indices == (),
                       "Non-scalar == or != is not allowed.")

        Operator.__init__(self, (condition, true_value, false_value))

    def evaluate(self, x, mapping, component, index_values):
        c = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        if c:
            a = self.ufl_operands[1]
        else:
            a = self.ufl_operands[2]
        return a.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "%s ? %s : %s" % tuple(parstr(o, self) for o in self.ufl_operands)


# --- Specific functions higher level than a conditional ---

@ufl_type(is_scalar=True, num_ops=1)
class MinValue(Operator):
    "UFL operator: Take the minimum of two values."
    __slots__ = ()

    def __init__(self, left, right):
        Operator.__init__(self, (left, right))
        ufl_assert(is_true_ufl_scalar(left) and is_true_ufl_scalar(right),
                   "Expecting scalar arguments.")

    def evaluate(self, x, mapping, component, index_values):
        a, b = self.ufl_operands
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        try:
            res = min(a, b)
        except ValueError:
            warning('Value error in evaluation of min() of %s and %s.' % self.ufl_operands)
            raise
        return res

    def __str__(self):
        return "min_value(%s, %s)" % self.ufl_operands


@ufl_type(is_scalar=True, num_ops=1)
class MaxValue(Operator):
    "UFL operator: Take the maximum of two values."
    __slots__ = ()

    def __init__(self, left, right):
        Operator.__init__(self, (left, right))
        ufl_assert(is_true_ufl_scalar(left) and is_true_ufl_scalar(right),
                   "Expecting scalar arguments.")

    def evaluate(self, x, mapping, component, index_values):
        a, b = self.ufl_operands
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        try:
            res = max(a, b)
        except ValueError:
            warning('Value error in evaluation of max() of %s and %s.' % self.ufl_operands)
            raise
        return res

    def __str__(self):
        return "max_value(%s, %s)" % self.ufl_operands
