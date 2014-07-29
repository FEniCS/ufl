"""Utility functions for checking properties of expressions."""

# Copyright (C) 2013-2014 Martin Sandve Alnes
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
# Modified by Anders Logg, 2008-2009

from ufl.expr import Expr

def is_python_scalar(expression):
    "Return True iff expression is of a Python scalar type."
    return isinstance(expression, (int, float))

def is_ufl_scalar(expression):
    """Return True iff expression is scalar-valued,
    but possibly containing free indices."""
    return isinstance(expression, Expr) and not expression.ufl_shape

def is_true_ufl_scalar(expression):
    """Return True iff expression is scalar-valued,
    with no free indices."""
    return isinstance(expression, Expr) and \
        not (expression.ufl_shape or expression.free_indices())

def is_globally_constant(expr):
    """Check if an expression is globally constant, which
    includes spatially independent constant coefficients that
    are not known before assembly time."""

    # TODO: This does not consider gradients of coefficients, so false negatives are possible.

    from ufl.algorithms.traversal import traverse_unique_terminals
    from ufl.argument import Argument
    from ufl.coefficient import Coefficient

    for e in traverse_unique_terminals(expr):
        if isinstance(e, Argument):
            return False
        if isinstance(e, Coefficient) and e.element().family() != "Real":
            return False
        if not e.is_cellwise_constant():
            return False

    # All terminals passed constant check
    return True

def is_scalar_constant_expression(expr):
    """Check if an expression is a globally constant scalar expression."""
    if is_python_scalar(expr):
        return True
    if expr.ufl_shape != ():
        return False
    return is_globally_constant(expr)
