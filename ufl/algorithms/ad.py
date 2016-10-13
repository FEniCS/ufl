# -*- coding: utf-8 -*-
"""Front-end for AD routines."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
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
# Modified by Anders Logg, 2009.

from ufl.log import warning
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives


def expand_derivatives(form, **kwargs):
    """Expand all derivatives of expr.

    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or CoefficientDerivative objects left, and Grad
    objects have been propagated to Terminal nodes.
    """
    # For a deprecation period (I see that dolfin-adjoint passes some
    # args here)
    if kwargs:
        warning("Deprecation: expand_derivatives no longer takes any keyword arguments")

    # Lower abstractions for tensor-algebra types into index notation
    form = apply_algebra_lowering(form)

    # Apply differentiation
    form = apply_derivatives(form)

    return form
