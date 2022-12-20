# -*- coding: utf-8 -*-
"""Front-end for AD routines."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009.

from warnings import warn
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives


def expand_derivatives(form):
    """Expand all derivatives of expr.

    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or CoefficientDerivative objects left, and Grad
    objects have been propagated to Terminal nodes.
    """
    # Lower abstractions for tensor-algebra types into index notation
    form = apply_algebra_lowering(form)

    # Apply differentiation
    form = apply_derivatives(form)

    return form
