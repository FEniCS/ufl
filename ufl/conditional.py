"""This module defines classes for conditional expressions."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import warnings

from ufl.checks import is_true_ufl_scalar
from ufl.constantvalue import Zero, as_ufl
from ufl.core.expr import ufl_err_str
from ufl.core.operator import Operator
from ufl.precedence import parstr

# --- Condition classes ---

# TODO: Would be nice with some kind of type system to show that this
# is a boolean type not a float type


class Condition(Operator):
    """Condition."""

    __slots__ = ()

    def __init__(self, operands):
        """Initialise."""
        Operator.__init__(self, operands)

    def __bool__(self):
        """Convert to a bool."""
        # Showing explicit error here to protect against misuse
        raise ValueError("UFL conditions cannot be evaluated as bool in a Python context.")

    __nonzero__ = __bool__


class BinaryCondition(Condition):
    """Binary condition."""

    __slots__ = ("_name",)

    def __init__(self, name, left, right):
        """Initialise."""
        left = as_ufl(left)
        right = as_ufl(right)

        Condition.__init__(self, (left, right))

        self._name = name

        if name in ("!=", "=="):
            # Since equals and not-equals are used for comparing
            # representations, we have to allow any shape here. The
            # scalar properties must be checked when used in
            # conditional instead!
            pass
        elif name in ("&&", "||"):
            # Binary operators acting on boolean expressions allow
            # only conditions
            for arg in (left, right):
                if not isinstance(arg, Condition):
                    raise ValueError(f"Expecting a Condition, not {ufl_err_str(arg)}.")
        else:
            # Binary operators acting on non-boolean expressions allow
            # only scalars
            if left.ufl_shape != () or right.ufl_shape != ():
                raise ValueError("Expecting scalar arguments.")
            if left.ufl_free_indices != () or right.ufl_free_indices != ():
                raise ValueError("Expecting scalar arguments.")

    def __str__(self):
        """Format as a string."""
        return "%s %s %s" % (
            parstr(self.ufl_operands[0], self),
            self._name,
            parstr(self.ufl_operands[1], self),
        )


# Not associating with __eq__, the concept of equality with == is
# reserved for object equivalence for use in set and dict.
class EQ(BinaryCondition):
    """Equality condition."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        BinaryCondition.__init__(self, "==", left, right)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a == b)

    def __bool__(self):
        """Convert to a bool."""
        return as_ufl(self.ufl_operands[0]) == as_ufl(self.ufl_operands[1])

    __nonzero__ = __bool__


# Not associating with __ne__, the concept of equality with == is
# reserved for object equivalence for use in set and dict.
class NE(BinaryCondition):
    """Not equal condition."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        BinaryCondition.__init__(self, "!=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a != b)

    def __bool__(self):
        """Convert to a bool."""
        return not as_ufl(self.ufl_operands[0]) == as_ufl(self.ufl_operands[1])

    __nonzero__ = __bool__


class LE(BinaryCondition):
    """Less than or equal condition."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        BinaryCondition.__init__(self, "<=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a <= b)


class GE(BinaryCondition):
    """Greater than or equal to condition."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        BinaryCondition.__init__(self, ">=", left, right)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a >= b)


class LT(BinaryCondition):
    """Less than condition."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        BinaryCondition.__init__(self, "<", left, right)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a < b)


class GT(BinaryCondition):
    """Greater than condition."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        BinaryCondition.__init__(self, ">", left, right)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a > b)


class AndCondition(BinaryCondition):
    """And condition."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        BinaryCondition.__init__(self, "&&", left, right)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a and b)


class OrCondition(BinaryCondition):
    """Or condition."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        BinaryCondition.__init__(self, "||", left, right)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        return bool(a or b)


class NotCondition(Condition):
    """Not condition."""

    __slots__ = ()

    def __init__(self, condition):
        """Initialise."""
        Condition.__init__(self, (condition,))
        if not isinstance(condition, Condition):
            raise ValueError("Expecting a condition.")

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return bool(not a)

    def __str__(self):
        """Format as a string."""
        return "!(%s)" % (str(self.ufl_operands[0]),)


class Conditional(Operator):
    """Conditional expression.

    In C++ these take the format `(condition ? true_value : false_value)`.
    """

    __slots__ = ()

    def __init__(self, condition, true_value, false_value):
        """Initialise."""
        if not isinstance(condition, Condition):
            raise ValueError("Expecting condition as first argument.")
        true_value = as_ufl(true_value)
        false_value = as_ufl(false_value)
        tsh = true_value.ufl_shape
        fsh = false_value.ufl_shape
        if tsh != fsh:
            raise ValueError("Shape mismatch between conditional branches.")
        tfi = true_value.ufl_free_indices
        ffi = false_value.ufl_free_indices
        if tfi != ffi:
            raise ValueError("Free index mismatch between conditional branches.")
        if isinstance(condition, (EQ, NE)):
            if not all(
                (
                    condition.ufl_operands[0].ufl_shape == (),
                    condition.ufl_operands[0].ufl_free_indices == (),
                    condition.ufl_operands[1].ufl_shape == (),
                    condition.ufl_operands[1].ufl_free_indices == (),
                )
            ):
                raise ValueError("Non-scalar == or != is not allowed.")

        Operator.__init__(self, (condition, true_value, false_value))

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        c = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        if c:
            a = self.ufl_operands[1]
        else:
            a = self.ufl_operands[2]
        return a.evaluate(x, mapping, component, index_values)

    def __str__(self):
        """Format as a string."""
        return "%s ? %s : %s" % tuple(parstr(o, self) for o in self.ufl_operands)

    def get_arity(self):
        """Get the arity."""
        from ufl.algorithms.check_arities import ArityMismatch, _afmt

        c = self.ufl_operands[0].get_arity()
        a = self.ufl_operands[1].get_arity()
        b = self.ufl_operands[2].get_arity()
        if c:
            raise ArityMismatch(f"Condition cannot depend on form arguments ({_afmt(a)}).")
        if a and isinstance(self.ufl_operands[2], Zero):
            # Allow conditional(c, arg, 0)
            return a
        elif b and isinstance(self.ufl_operands[1], Zero):
            # Allow conditional(c, 0, arg)
            return b
        elif a == b:
            # Allow conditional(c, test, test)
            return a
        else:
            # Do not allow e.g. conditional(c, test, trial),
            # conditional(c, test, nonzeroconstant)
            raise ArityMismatch(
                "Conditional subexpressions with non-matching form arguments "
                f"{_afmt(a)} vs {_afmt(b)}."
            )


# --- Specific functions higher level than a conditional ---


class MinValue(Operator):
    """Take the minimum of two values."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        Operator.__init__(self, (left, right))
        if not (is_true_ufl_scalar(left) and is_true_ufl_scalar(right)):
            raise ValueError("Expecting scalar arguments.")

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a, b = self.ufl_operands
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        try:
            res = min(a, b)
        except ValueError:
            warnings.warn("Value error in evaluation of min() of %s and %s." % self.ufl_operands)
            raise
        return res

    def __str__(self):
        """Format as a string."""
        return "min_value(%s, %s)" % self.ufl_operands


class MaxValue(Operator):
    """Take the maximum of two values."""

    __slots__ = ()

    def __init__(self, left, right):
        """Initialise."""
        Operator.__init__(self, (left, right))
        if not (is_true_ufl_scalar(left) and is_true_ufl_scalar(right)):
            raise ValueError("Expecting scalar arguments.")

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a, b = self.ufl_operands
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        try:
            res = max(a, b)
        except ValueError:
            warnings.warn("Value error in evaluation of max() of %s and %s." % self.ufl_operands)
            raise
        return res

    def __str__(self):
        """Format as a string."""
        return "max_value(%s, %s)" % self.ufl_operands
