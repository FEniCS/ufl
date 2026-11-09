"""Front-end for AD routines."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009.

import warnings

from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_coefficient_derivatives, apply_derivatives
from ufl.classes import BaseForm, Derivative, Expr


def expand_derivatives(expr: Expr | BaseForm, **kwargs):
    """Expand all derivatives of expr.

    In the returned expression, which is mathematically
    equivalent to expr, there are no VariableDerivative
    or CoefficientDerivative objects left, and Grad
    objects have been propagated to Terminal nodes.
    """
    # For a deprecation period (I see that dolfin-adjoint passes some
    # args here)
    if kwargs:
        warnings.warn("Deprecation: expand_derivatives no longer takes any keyword arguments")

    # Lower abstractions for tensor-algebra types into index notation
    expr = apply_algebra_lowering(expr)

    # Apply differentiation
    expr = apply_derivatives(expr)

    return expr


def expand_coefficient_derivatives(expr: Expr | BaseForm) -> Expr | BaseForm:
    """Expand coefficient and BaseForm derivatives, preserving other derivatives.

    Spatial derivatives are left on compound expressions for the form compiler.
    Tensor algebra is lowered to index notation before differentiation.

    Args:
        expr: The expression whose coefficient derivatives are to be expanded.

    Returns:
        The expression with coefficient and BaseForm derivatives expanded.
    """
    # Lower abstractions for tensor-algebra types into index notation
    expr = apply_algebra_lowering(expr, preserve_types=(Derivative,))

    # Apply differentiation
    expr = apply_coefficient_derivatives(expr)

    return expr
