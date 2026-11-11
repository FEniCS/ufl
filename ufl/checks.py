"""Utility functions for checking properties of expressions."""

# Copyright (C) 2013-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009

from ufl.core.expr import Expr
from ufl.core.terminal import FormArgument
from ufl.corealg.traversal import cutoff_unique_post_traversal, traverse_unique_terminals
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


def is_scalar_constant_expression(expr):
    """Check if an expression is a globally constant scalar expression."""
    if is_python_scalar(expr):
        return True
    if not isinstance(expr, Expr) or expr.ufl_shape:
        return False

    from ufl.core.base_form_operator import BaseFormOperator  # Avoid circular import

    # TODO: This does not consider gradients of coefficients, so false
    # negatives are possible.
    cutoff_types = [issubclass(cls, BaseFormOperator) for cls in Expr._ufl_all_classes_]
    for e in cutoff_unique_post_traversal(expr, cutoff_types):
        if isinstance(e, FormArgument | BaseFormOperator):
            # Arguments, Coefficients and base form operators all take values in
            # their function space, so accept only globally constant spaces.
            # The operands of a base form operator do not matter, and an
            # operator without a function space is a 0-form, not an expression.
            space = e.ufl_function_space()
            if space is None:
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
