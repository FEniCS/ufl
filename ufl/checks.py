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
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.geometry import GeometricQuantity
from ufl.sobolevspace import H1


def is_python_scalar(expression):
    """Return True iff expression is of a Python scalar type."""
    return isinstance(expression, (int, float, complex))


def is_ufl_scalar(expression):
    """Return True iff expression is scalar-valued, but possibly containing free indices."""
    return isinstance(expression, Expr) and not expression.ufl_shape


def is_true_ufl_scalar(expression):
    """Return True iff expression is scalar-valued, with no free indices."""
    return isinstance(expression, Expr) and not (expression.ufl_shape or expression.ufl_free_indices)


def is_cellwise_constant(expr):
    """Return whether expression is constant over a single cell."""
    # TODO: Implement more accurately considering e.g. derivatives?
    return all(e.is_cellwise_constant() for e in traverse_unique_terminals(expr))


def is_scalar_constant_expression(expr):
    """Check if an expression is a globally constant scalar expression."""
    if is_python_scalar(expr):
        return True
    if expr.ufl_shape:
        return False

    # TODO: This does not consider gradients of coefficients, so false
    # negatives are possible.
    for e in traverse_unique_terminals(expr):
        # Return False if any single terminal is not constant
        if isinstance(e, FormArgument):
            # Accept only globally constant Arguments and Coefficients
            if e.ufl_element().embedded_superdegree > 0 or e.ufl_element() not in H1:
                return False
        elif isinstance(e, GeometricQuantity):
            # Reject all geometric quantities, they all vary over
            # cells
            return False

    # All terminals passed constant check
    return True
