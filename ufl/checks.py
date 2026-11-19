"""Utility functions for checking properties of expressions."""

# Copyright (C) 2013-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009

from functools import cache

from ufl.core.expr import Expr
from ufl.core.terminal import FormArgument
from ufl.corealg.traversal import (
    cutoff_unique_post_traversal,
    traverse_unique_terminals,
    unique_pre_traversal,
)
from ufl.duals import is_dual
from ufl.geometry import GeometricQuantity
from ufl.sobolevspace import H1


def is_python_scalar(expression):
    """Return True iff expression is of a Python scalar type."""
    return isinstance(expression, int | float | complex)


def is_ufl_scalar(expression):
    """Return True iff expression is scalar-valued, but possibly containing free indices."""
    return isinstance(expression, Expr) and not expression.ufl_shape


def is_true_ufl_scalar(expression):
    """Return True iff expression is scalar-valued, with no free indices."""
    return isinstance(expression, Expr) and not (
        expression.ufl_shape or expression.ufl_free_indices
    )


def is_cellwise_constant(expr):
    """Return whether expression is constant over a single cell."""
    # TODO: Implement more accurately considering e.g. derivatives?
    return all(e.is_cellwise_constant() for e in traverse_unique_terminals(expr))


@cache
def _base_form_operator_types(num_types: int) -> tuple[bool, ...]:
    """Flag the UFL types that are base form operators, indexed by type code.

    The cache key is the number of registered types, so that types registered
    later, such as downstream subclasses, refresh it.
    """
    from ufl.core.base_form_operator import BaseFormOperator  # Avoid circular import

    return tuple(issubclass(cls, BaseFormOperator) for cls in Expr._ufl_all_classes_[:num_types])


def is_scalar_constant_expression(expr):
    """Check if an expression is a globally constant scalar expression."""
    if is_python_scalar(expr):
        return True
    if not isinstance(expr, Expr) or expr.ufl_shape:
        return False

    from ufl.core.base_form_operator import BaseFormOperator  # Avoid circular import

    # TODO: This does not consider gradients of coefficients, so false
    # negatives are possible.
    cutoff_types = _base_form_operator_types(len(Expr._ufl_all_classes_))
    for e in cutoff_unique_post_traversal(expr, cutoff_types):
        if isinstance(e, FormArgument | BaseFormOperator):
            # Arguments, Coefficients and base form operators all take values in
            # their function space, so accept only globally constant spaces.
            # The operands of a base form operator do not matter, and an
            # operator without a function space or in a dual space is a form.
            space = e.ufl_function_space()
            if space is None or is_dual(space):
                return False
            element = space.ufl_element()
            if element.embedded_superdegree > 0 or element not in H1:
                return False
        elif isinstance(e, GeometricQuantity):
            # Reject all geometric quantities, they all vary over
            # cells
            return False

    # All nodes passed constant check
    return True


def _is_scaling_constant(expr):
    """Check if an expression is a number or a scalar expression of Constants.

    Unlike ``is_scalar_constant_expression``, this rejects functions in a Real space,
    which a form acts on when it is multiplied by them on the right.
    """
    if not is_scalar_constant_expression(expr):
        return False
    if not isinstance(expr, Expr):
        return True
    from ufl.form import BaseForm  # Avoid circular import

    return not any(isinstance(e, FormArgument | BaseForm) for e in unique_pre_traversal(expr))
