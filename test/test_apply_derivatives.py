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
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering


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
@pytest.mark.parametrize("order", [2])
@pytest.mark.parametrize("lower_alg", [True, False])
@pytest.mark.parametrize("lower_geo", [True, False])
@pytest.mark.parametrize("apply_deriv", [True, False])
def test_diff_grad_jacobian_zero(cell, gdim, order, lower_alg, lower_geo, apply_deriv):
    tdim = cell.topological_dimension()

    domain = Mesh(LagrangeElement(cell, order, (gdim,)))

    J = Jacobian(domain)
    assert J.ufl_shape == (gdim, tdim)

    F = grad(J)
    if lower_alg:
        F = apply_algebra_lowering(F)

    if lower_geo:
        F = apply_geometry_lowering(F)

    if apply_deriv:
        F = apply_derivatives(F)

    V = FunctionSpace(domain, LagrangeElement(cell, 1))
    u = Coefficient(V)

    δF_u = diff(F, u)

    if lower_alg:
        δF_u = apply_algebra_lowering(δF_u)

    if lower_geo:
        δF_u = apply_geometry_lowering(δF_u)

    δF_u = apply_derivatives(δF_u)

    assert δF_u == 0
    assert δF_u.ufl_shape == (gdim, tdim, gdim)
