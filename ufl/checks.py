"""Utility functions for checking properties of expressions."""

# Copyright (C) 2013-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009

from ufl.core.expr import Expr
from ufl.corealg.traversal import traverse_unique_terminals


def is_python_scalar(expression):
    "Return True iff expression is of a Python scalar type."
    return isinstance(expression, (int, float, complex))


def is_ufl_scalar(expression):
    """Return True iff expression is scalar-valued,
    but possibly containing free indices."""
    return isinstance(expression, Expr) and not expression.ufl_shape


def is_true_ufl_scalar(expression):
    """Return True iff expression is scalar-valued,
    with no free indices."""
    return isinstance(expression, Expr) and not (expression.ufl_shape or expression.ufl_free_indices)


def is_cellwise_constant(expr):
    "Return whether expression is constant over a single cell."
    # TODO: Implement more accurately considering e.g. derivatives?
    return all(t.is_cellwise_constant() for t in traverse_unique_terminals(expr))


def is_globally_constant(expr):
    """Check if an expression is globally constant, which
    includes spatially independent constant coefficients that
    are not known before assembly time."""
    # TODO: This does not consider gradients of coefficients, so false
    # negatives are possible.
    # from ufl.argument import Argument
    # from ufl.coefficient import Coefficient
    from ufl.core.terminal import FormArgument
    from ufl.geometry import GeometricQuantity
    for e in traverse_unique_terminals(expr):
        # Return False if any single terminal is not constant
        if e._ufl_is_literal_:
            # Accept literals first, they are the most common
            # terminals
            continue
        elif isinstance(e, FormArgument):
            # Accept only Real valued Arguments and Coefficients
            if e.ufl_element()._is_globally_constant():
                continue
            else:
                return False
        elif isinstance(e, GeometricQuantity):
            # Reject all geometric quantities, they all vary over
            # cells
            return False

    # All terminals passed constant check
    return True


def is_scalar_constant_expression(expr):
    """Check if an expression is a globally constant scalar expression."""
    if is_python_scalar(expr):
        return True
    if expr.ufl_shape:
        return False
    return is_globally_constant(expr)
