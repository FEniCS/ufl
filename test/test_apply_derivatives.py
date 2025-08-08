# Copyright (C) 2025 Paul T. Kühner
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from utils import LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Jacobian,
    Mesh,
    diff,
    grad,
    interval,
    tetrahedron,
    triangle,
)
from ufl.algorithms.apply_derivatives import apply_derivatives


@pytest.mark.parametrize(
    "cell,gdim",
    [
        (interval, 1),
        (interval, 2),
        (interval, 3),
        (triangle, 2),
        (triangle, 3),
        (tetrahedron, 3),
    ],
)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_diff_grad_jacobian_zero(cell, gdim, order):
    tdim = cell.topological_dimension()

    domain = Mesh(LagrangeElement(cell, order, (gdim,)))

    J0 = Jacobian(domain)
    assert J0.ufl_shape == (gdim, tdim)

    J = grad(J0)

    V = FunctionSpace(domain, LagrangeElement(cell, 1))
    u = Coefficient(V)

    δJ_u = diff(J, u)
    δJ_u = apply_derivatives(δJ_u)

    assert δJ_u == 0
    assert δJ_u.ufl_shape == (gdim, tdim, gdim)
