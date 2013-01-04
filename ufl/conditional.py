"""This module defines classes for conditional expressions."""

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
# First added:  2008-08-20
# Last changed: 2012-11-30

from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.operatorbase import Operator
from ufl.constantvalue import as_ufl
from ufl.precedence import parstr
from ufl.exprequals import expr_equals

#--- Condition classes ---

class Condition(Operator):
    __slots__ = ()
    def __init__(self):
        Operator.__init__(self)

    # Functions like these are an indication that a better type system could be useful:

    def free_indices(self):
        error("Calling free_indices on Condition is an error.")

    def index_dimensions(self):
        error("Calling index_dimensions on Condition is an error.")

    def shape(self):
        error("Calling shape on Condition is an error.")

    def __nonzero__(self):
        # Showing explicit error here to protect against misuse
        error("UFL conditions cannot be evaluated as bool in a Python context.")
        #return NotImplemented

class BinaryCondition(Condition):
    __slots__ = ('_name', '_left', '_right',)
    def __init__(self, name, left, right):
        Operator.__init__(self)
        self._name = name
        self._left = as_ufl(left)
        self._right = as_ufl(right)
        if name in ('!=', '=='):
            # Since equals and not-equals are used for comparing representations,
            # we have to allow any shape here. The scalar properties must be
            # checked when used in conditional instead!
            pass
        elif name in ('&&', '||'):
            # Binary operators acting on boolean expressions allow only conditions
            ufl_assert(isinstance(self._left, Condition),
                       "Expecting a Condition, not a %s." % self._left._uflclass)
            ufl_assert(isinstance(self._right, Condition),
                       "Expecting a Condition, not a %s." % self._right._uflclass)
        else:
            # Binary operators acting on non-boolean expressions allow only scalars
            ufl_assert(self._left.shape() == () \
                           and  self._right.shape() == (),
                       "Expecting scalar arguments.")
            ufl_assert(self._left.free_indices() == () \
                           and self._right.free_indices() == (),
                       "Expecting scalar arguments.")

    def operands(self):
        # A BinaryCondition should never be constructed directly,
        # so these two arguments correspond to the constructor
        # arguments of the subclasses EQ etc.
        return (self._left, self._right)

    def __str__(self):
        return "%s %s %s" % (parstr(self._left, self), self._name, parstr(self._right, self))

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self._left, self._right)

class EQ(BinaryCondition):
    __slots__ = ()
    def __init__(self, left, right):
        BinaryCondition.__init__(self, "==", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return bool(a == b)

    def __nonzero__(self):
        return expr_equals(self._left, self._right)

class NE(BinaryCondition):
    __slots__ = ()
    def __init__(self, left, right):
        BinaryCondition.__init__(self, "!=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return bool(a != b)

    def __nonzero__(self):
        return not expr_equals(self._left, self._right)

class LE(BinaryCondition):
    __slots__ = ()
    def __init__(self, left, right):
        BinaryCondition.__init__(self, "<=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return bool(a <= b)

class GE(BinaryCondition):
    __slots__ = ()
    def __init__(self, left, right):
        BinaryCondition.__init__(self, ">=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return bool(a >= b)

class LT(BinaryCondition):
    __slots__ = ()
    def __init__(self, left, right):
        BinaryCondition.__init__(self, "<", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return bool(a < b)

class GT(BinaryCondition):
    __slots__ = ()
    def __init__(self, left, right):
        BinaryCondition.__init__(self, ">", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return bool(a > b)

class AndCondition(BinaryCondition):
    __slots__ = ()
    def __init__(self, left, right):
        BinaryCondition.__init__(self, "&&", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return bool(a and b)

class OrCondition(BinaryCondition):
    __slots__ = ()
    def __init__(self, left, right):
        BinaryCondition.__init__(self, "||", left, right)

    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return bool(a or b)

class NotCondition(Condition):
    __slots__ = ('_condition',)
    def __init__(self, condition):
        Condition.__init__(self)
        ufl_assert(isinstance(condition, Condition), "Expecting a condition.")
        self._condition = condition

    def operands(self):
        return (self._condition,)

    def evaluate(self, x, mapping, component, index_values):
        a = self._condition.evaluate(x, mapping, component, index_values)
        return bool(not a)

    def __str__(self):
        return "!(%s)" % (str(self._condition),)

    def __repr__(self):
        return "NotCondition(%r)" % (self._condition,)

#--- Conditional expression (condition ? true_value : false_value) ---

class Conditional(Operator):
    __slots__ = ("_condition", "_true_value", "_false_value",)
    def __init__(self, condition, true_value, false_value):
        Operator.__init__(self)
        ufl_assert(isinstance(condition, Condition), "Expectiong condition as first argument.")
        true_value = as_ufl(true_value)
        false_value = as_ufl(false_value)
        tsh = true_value.shape()
        fsh = false_value.shape()
        ufl_assert(tsh == fsh, "Shape mismatch between conditional branches.")
        tfi = true_value.free_indices()
        ffi = false_value.free_indices()
        ufl_assert(tfi == ffi, "Free index mismatch between conditional branches.")
        if isinstance(condition, (EQ,NE)):
            ufl_assert(condition._left.shape() == ()
                       and condition._left.free_indices() == ()
                       and condition._right.shape() == ()
                       and condition._right.free_indices() == (),
                       "Non-scalar == or != is not allowed.")
        self._condition = condition
        self._true_value = true_value
        self._false_value = false_value

    def operands(self):
        return (self._condition, self._true_value, self._false_value)

    def free_indices(self):
        return self._true_value.free_indices()

    def index_dimensions(self):
        return self._true_value.index_dimensions()

    def shape(self):
        return self._true_value.shape()

    def evaluate(self, x, mapping, component, index_values):
        c = self._condition.evaluate(x, mapping, component, index_values)
        if c:
            a = self._true_value
        else:
            a = self._false_value
        return a.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "%s ? %s : %s" % tuple(parstr(o, self) for o in self.operands())

    def __repr__(self):
        return "Conditional(%r, %r, %r)" % self.operands()

