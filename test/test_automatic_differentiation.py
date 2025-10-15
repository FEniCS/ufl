# Copyright (C) 2014-2025 Martin Sandve Alnæs and Paul T. Kühner
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Automatic differentiation tests.

These tests should cover the behaviour of the automatic differentiation
algorithm at a technical level, and are thus implementation specific.
Other tests check for mathematical correctness of diff and derivative.
"""

import pytest
from utils import FiniteElement, LagrangeElement

from ufl import (
    And,
    Argument,
    CellDiameter,
    CellVolume,
    Circumradius,
    Coefficient,
    Constant,
    FacetArea,
    FacetNormal,
    FunctionSpace,
    Identity,
    Jacobian,
    JacobianDeterminant,
    JacobianInverse,
    MaxCellEdgeLength,
    MaxFacetEdgeLength,
    Mesh,
    MinCellEdgeLength,
    MinFacetEdgeLength,
    Not,
    Or,
    PermutationSymbol,
    SpatialCoordinate,
    acos,
    as_matrix,
    as_tensor,
    as_ufl,
    as_vector,
    asin,
    atan,
    bessel_I,
    bessel_J,
    bessel_K,
    bessel_Y,
    cofac,
    conditional,
    cos,
    cross,
    derivative,
    det,
    dev,
    diff,
    dot,
    eq,
    erf,
    exp,
    ge,
    grad,
    gt,
    indices,
    inner,
    interval,
    inv,
    le,
    ln,
    lt,
    ne,
    outer,
    replace,
    sin,
    skew,
    sqrt,
    sym,
    tan,
    tetrahedron,
    tr,
    triangle,
    variable,
)
from ufl.algorithms import expand_derivatives
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.conditional import Conditional
from ufl.corealg.traversal import unique_post_traversal
from ufl.pullback import identity_pullback
from ufl.sobolevspace import L2


class ExpressionCollection:
    def __init__(self, cell, geometry_degree, gdim):
        self.cell = cell
        domain = Mesh(LagrangeElement(cell, geometry_degree, (gdim,)))

        x = SpatialCoordinate(domain)
        n = FacetNormal(domain)
        c = CellVolume(domain)
        R = Circumradius(domain)
        h = CellDiameter(domain)
        f = FacetArea(domain)
        # s = CellSurfaceArea(domain)
        mince = MinCellEdgeLength(domain)
        maxce = MaxCellEdgeLength(domain)
        minfe = MinFacetEdgeLength(domain)
        maxfe = MaxFacetEdgeLength(domain)
        J = Jacobian(domain)
        detJ = JacobianDeterminant(domain)
        invJ = JacobianInverse(domain)
        # FIXME: Add all new geometry types here!

        ident = Identity(gdim)
        eps = PermutationSymbol(gdim)

        U = FiniteElement("Undefined", cell, None, (), identity_pullback, L2)
        V = FiniteElement("Undefined", cell, None, (gdim,), identity_pullback, L2)
        W = FiniteElement("Undefined", cell, None, (gdim, gdim), identity_pullback, L2)

        u_space = FunctionSpace(domain, U)
        v_space = FunctionSpace(domain, V)
        w_space = FunctionSpace(domain, W)

        u = Coefficient(u_space)
        v = Coefficient(v_space)
        w = Coefficient(w_space)
        du = Argument(u_space, 0)
        dv = Argument(v_space, 1)
        dw = Argument(w_space, 2)

        class ObjectCollection:
            pass

        self.shared_objects = ObjectCollection()
        for key, value in list(locals().items()):
            setattr(self.shared_objects, key, value)

        self.literals = list(map(as_ufl, [0, 1, 3.14, ident, eps]))
        self.geometry = [x, n, c, R, h, f, mince, maxce, minfe, maxfe, J, detJ, invJ]
        self.functions = [u, du, v, dv, w, dw]

        self.terminals = []
        self.terminals += self.literals
        self.terminals += self.geometry
        self.terminals += self.functions

        self.algebra = [
            u * 2,
            v * 2,
            w * 2,
            u + 2 * u,
            v + 2 * v,
            w + 2 * w,
            2 / u,
            u / 2,
            v / 2,
            w / 2,
            u**3,
            3**u,
        ]
        self.mathfunctions = [
            abs(u),
            sqrt(u),
            exp(u),
            ln(u),
            cos(u),
            sin(u),
            tan(u),
            acos(u),
            asin(u),
            atan(u),
            erf(u),
            bessel_I(1, u),
            bessel_J(1, u),
            bessel_K(1, u),
            bessel_Y(1, u),
        ]
        self.variables = [
            variable(u),
            variable(v),
            variable(w),
            variable(w * u),
            3 * variable(w * u),
        ]

        if gdim == 1:
            w2 = as_matrix(((u**2,),))
        if gdim == 2:
            w2 = as_matrix(((u**2, u**3), (u**4, u**5)))
        if gdim == 3:
            w2 = as_matrix(((u**2, u**3, u**4), (u**4, u**5, u**6), (u**6, u**7, u**8)))

        # Indexed,  ListTensor, ComponentTensor, IndexSum
        i, j, k, l = indices(4)  # noqa: E741
        self.indexing = [
            v[0],
            w[gdim - 1, 0],
            v[i],
            w[i, j],
            v[:],
            w[0, :],
            w[:, 0],
            v[...],
            w[0, ...],
            w[..., 0],
            v[i] * v[j],
            w[i, 0] * v[j],
            w[gdim - 1, j] * v[i],
            v[i] * v[i],
            w[i, 0] * w[0, i],
            v[i] * w[0, i],
            v[j] * w[gdim - 1, j],
            w[i, i],
            w[i, j] * w[j, i],
            as_tensor(v[i] * w[k, 0], (k, i)),
            as_tensor(v[i] * w[k, 0], (k, i))[:, l],
            as_tensor(w[i, j] * w[k, l], (k, j, l, i)),
            as_tensor(w[i, j] * w[k, l], (k, j, l, i))[0, 0, 0, 0],
            as_vector((u, 2, 3)),
            as_matrix(((u**2, u**3), (u**4, u**5))),
            as_vector((u, 2, 3))[i],
            w2[i, j] * w[i, j],
        ]
        self.conditionals = [
            conditional(le(u, 1.0), 1, 0),
            conditional(eq(3.0, u), 1, 0),
            conditional(ne(sin(u), cos(u)), 1, 0),
            conditional(lt(sin(u), cos(u)), 1, 0),
            conditional(ge(sin(u), cos(u)), 1, 0),
            conditional(gt(sin(u), cos(u)), 1, 0),
            conditional(And(lt(u, 3), gt(u, 1)), 1, 0),
            conditional(Or(lt(u, 3), gt(u, 1)), 1, 0),
            conditional(Not(ge(u, 0.0)), 1, 0),
            conditional(le(u, 0.0), 1, 2),
            conditional(Not(ge(u, 0.0)), 1, 2),
            conditional(And(Not(ge(u, 0.0)), lt(u, 1.0)), 1, 2),
            conditional(le(u, 0.0), u**3, ln(u)),
        ]
        self.restrictions = [u("+"), u("-"), v("+"), v("-"), w("+"), w("-")]
        if gdim > 1:
            i, j = indices(2)
            self.restrictions += [
                v("+")[i] * v("+")[i],
                v[i]("+") * v[i]("+"),
                (v[i] * v[i])("+"),
                (v[i] * v[j])("+") * w[i, j]("+"),
            ]

        self.noncompounds = []
        self.noncompounds += self.algebra
        self.noncompounds += self.mathfunctions
        self.noncompounds += self.variables
        self.noncompounds += self.indexing
        self.noncompounds += self.conditionals
        self.noncompounds += self.restrictions

        if gdim == 1:
            self.tensorproducts = []
        else:
            self.tensorproducts = [
                dot(v, v),
                dot(v, w),
                dot(w, w),
                inner(v, v),
                inner(w, w),
                outer(v, v),
                outer(w, v),
                outer(v, w),
                outer(w, w),
            ]

        if gdim == 1:
            self.tensoralgebra = []
        else:
            self.tensoralgebra = [
                w.T,
                sym(w),
                skew(w),
                dev(w),
                det(w),
                tr(w),
                cofac(w),
                inv(w),
            ]

        if gdim != 3:
            self.crossproducts = []
        else:
            self.crossproducts = [
                cross(v, v),
                cross(v, 2 * v),
                cross(v, w[0, :]),
                cross(v, w[:, 1]),
                cross(w[:, 0], v),
            ]

        self.compounds = []
        self.compounds += self.tensorproducts
        self.compounds += self.tensoralgebra
        self.compounds += self.crossproducts

        self.all_expressions = []
        self.all_expressions += self.terminals
        self.all_expressions += self.noncompounds
        self.all_expressions += self.compounds


@pytest.fixture(
    params=[
        (interval, 1, 1),
        (interval, 2, 1),
        (interval, 3, 1),
        (interval, 1, 2),
        (interval, 2, 2),
        (interval, 3, 2),
        (interval, 1, 3),
        (interval, 2, 3),
        (interval, 3, 3),
        (triangle, 1, 2),
        (triangle, 2, 2),
        (triangle, 3, 2),
        (triangle, 1, 3),
        (triangle, 2, 3),
        (triangle, 3, 3),
        (tetrahedron, 1, 3),
        (tetrahedron, 2, 3),
        (tetrahedron, 3, 3),
    ]
)
def d_expr(request):
    cell, geometry_degree, gdim = request.param
    expr = ExpressionCollection(cell, geometry_degree, gdim)
    return cell, expr


def _test_no_derivatives_no_change(self, collection):
    for expr in collection:
        before = expr
        after = expand_derivatives(before)
        # print '\n', str(before), '\n', str(after), '\n'
        self.assertEqualTotalShape(before, after)
        assert before == after


def _test_no_derivatives_but_still_changed(self, collection):
    # Planning to fix these:
    for expr in collection:
        before = expr
        after = expand_derivatives(before)
        # print '\n', str(before), '\n', str(after), '\n'
        self.assertEqualTotalShape(before, after)
        # assert before == after # Without expand_compounds
        self.assertNotEqual(before, after)  # With expand_compounds


def test_only_terminals_no_change(self, d_expr):
    _d, ex = d_expr
    _test_no_derivatives_no_change(self, ex.terminals)


def test_no_derivatives_no_change(self, d_expr):
    _d, ex = d_expr
    _test_no_derivatives_no_change(self, ex.noncompounds)


def xtest_compounds_no_derivatives_no_change(
    self, d_expr
):  # This test fails with expand_compounds enabled
    _d, ex = d_expr
    _test_no_derivatives_no_change(self, ex.compounds)


def test_zero_derivatives_of_terminals_produce_the_right_types_and_shapes(self, d_expr):
    _d, ex = d_expr
    _test_zero_derivatives_of_terminals_produce_the_right_types_and_shapes(self, ex)


def _test_zero_derivatives_of_terminals_produce_the_right_types_and_shapes(self, collection):
    c = Constant(collection.shared_objects.domain)

    u = Coefficient(collection.shared_objects.u_space)
    v = Coefficient(collection.shared_objects.v_space)
    w = Coefficient(collection.shared_objects.w_space)

    for t in collection.terminals:
        for var in (u, v, w):
            before = derivative(t, var)  # This will often get preliminary simplified to zero
            after = expand_derivatives(before)
            expected = 0 * t
            # print '\n', str(expected), '\n', str(after), '\n', str(before), '\n'
            assert after == expected

            before = derivative(c * t, var)  # This will usually not get simplified to zero
            after = expand_derivatives(before)
            expected = 0 * t
            # print '\n', str(expected), '\n', str(after), '\n', str(before), '\n'
            assert after == expected


def test_zero_diffs_of_terminals_produce_the_right_types_and_shapes(self, d_expr):
    _d, ex = d_expr
    _test_zero_diffs_of_terminals_produce_the_right_types_and_shapes(self, ex)


def _test_zero_diffs_of_terminals_produce_the_right_types_and_shapes(self, collection):
    c = Constant(collection.shared_objects.domain)

    u = Coefficient(collection.shared_objects.u_space)
    v = Coefficient(collection.shared_objects.v_space)
    w = Coefficient(collection.shared_objects.w_space)

    vu = variable(u)
    vv = variable(v)
    vw = variable(w)
    for t in collection.terminals:
        for var in (vu, vv, vw):
            before = diff(t, var)  # This will often get preliminary simplified to zero
            after = expand_derivatives(before)
            expected = 0 * outer(t, var)
            assert after == expected

            before = diff(c * t, var)  # This will usually not get simplified to zero
            after = expand_derivatives(before)
            expected = 0 * outer(t, var)
            assert after == expected


def test_zero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, d_expr):
    _d, ex = d_expr
    _test_zero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, ex)


def _test_zero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, collection):
    u = Coefficient(collection.shared_objects.u_space)
    v = Coefficient(collection.shared_objects.v_space)
    w = Coefficient(collection.shared_objects.w_space)

    # for t in chain(collection.noncompounds, collection.compounds):
    for t in collection.noncompounds:
        for var in (u, v, w):
            before = derivative(t, var)
            after = expand_derivatives(before)
            expected = 0 * t

            assert after == expected


def test_zero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, d_expr):
    _d, ex = d_expr
    _test_zero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, ex)


def _test_zero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, collection):
    u = Coefficient(collection.shared_objects.u_space)
    v = Coefficient(collection.shared_objects.v_space)
    w = Coefficient(collection.shared_objects.w_space)

    vu = variable(u)
    vv = variable(v)
    vw = variable(w)

    # for t in chain(collection.noncompounds, collection.compounds):
    for t in collection.noncompounds:
        for var in (vu, vv, vw):
            before = diff(t, var)
            after = expand_derivatives(before)
            expected = 0 * outer(t, var)

            assert after == expected


def test_nonzero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, d_expr):
    _d, ex = d_expr
    _test_nonzero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, ex)


def _test_nonzero_derivatives_of_noncompounds_produce_the_right_types_and_shapes(self, collection):
    u = collection.shared_objects.u
    v = collection.shared_objects.v
    w = collection.shared_objects.w

    # for t in chain(collection.noncompounds, collection.compounds):
    for t in collection.noncompounds:
        for var in (u, v, w):
            # Include d/dx [z ? y: x] but not d/dx [x ? f: z]
            if isinstance(t, Conditional) and (var in unique_post_traversal(t.ufl_operands[0])):
                continue

            before = derivative(t, var)
            after = expand_derivatives(before)
            expected_shape = 0 * t

            if var in unique_post_traversal(t):
                self.assertEqualTotalShape(after, expected_shape)
                self.assertNotEqual(after, expected_shape)
            else:
                assert after == expected_shape


def test_nonzero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, d_expr):
    _d, ex = d_expr
    _test_nonzero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, ex)


def _test_nonzero_diffs_of_noncompounds_produce_the_right_types_and_shapes(self, collection):
    u = collection.shared_objects.u
    v = collection.shared_objects.v
    w = collection.shared_objects.w

    vu = variable(u)
    vv = variable(v)
    vw = variable(w)

    # for t in chain(collection.noncompounds, collection.compounds):
    for t in collection.noncompounds:
        t = replace(t, {u: vu, v: vv, w: vw})
        for var in (vu, vv, vw):
            # Include d/dx [z ? y: x] but not d/dx [x ? f: z]
            if isinstance(t, Conditional) and (var in unique_post_traversal(t.ufl_operands[0])):
                continue

            before = diff(t, var)
            after = expand_derivatives(before)
            expected_shape = 0 * outer(t, var)  # expected shape, not necessarily value

            if var in unique_post_traversal(t):
                self.assertEqualTotalShape(after, expected_shape)
                self.assertNotEqual(after, expected_shape)
            else:
                assert after == expected_shape


def test_grad_coeff(self, d_expr):
    _d, collection = d_expr

    u = collection.shared_objects.u
    v = collection.shared_objects.v
    w = collection.shared_objects.w
    for f in (u, v, w):
        before = grad(f)
        after = expand_derivatives(before)

        self.assertEqualTotalShape(before, after)
        if f is u:  # Differing by being wrapped in indexing types
            assert before == after

        before = grad(grad(f))
        after = expand_derivatives(before)
        self.assertEqualTotalShape(before, after)
        # assert before == after # Differing by being wrapped in indexing types

        before = grad(grad(grad(f)))
        after = expand_derivatives(before)
        self.assertEqualTotalShape(before, after)
        # assert before == after # Differing by being wrapped in indexing types


def test_derivative_grad_coeff(self, d_expr):
    _d, collection = d_expr

    u = collection.shared_objects.u
    v = collection.shared_objects.v
    w = collection.shared_objects.w
    for f in (u, v, w):
        before = derivative(grad(f), f)
        after = expand_derivatives(before)
        self.assertEqualTotalShape(before, after)
        # assert after == expected

        before = derivative(grad(grad(f)), f)
        after = expand_derivatives(before)
        self.assertEqualTotalShape(before, after)
        # assert after == expected

        before = derivative(grad(grad(grad(f))), f)
        after = expand_derivatives(before)
        self.assertEqualTotalShape(before, after)
        # assert after == expected


def xtest_derivative_grad_coeff_with_variation_components(self, d_expr):
    _d, collection = d_expr

    v = collection.shared_objects.v
    w = collection.shared_objects.w
    dv = collection.shared_objects.dv
    dw = collection.shared_objects.dw
    for g, dg in ((v, dv), (w, dw)):
        # Pick a single component
        ii = (0,) * (len(g.ufl_shape))
        f = g[ii]
        df = dg[ii]

        before = derivative(grad(g), f, df)
        after = expand_derivatives(before)
        self.assertEqualTotalShape(before, after)
        # assert after == expected

        before = derivative(grad(grad(g)), f, df)
        after = expand_derivatives(before)
        self.assertEqualTotalShape(before, after)
        # assert after == expected

        before = derivative(grad(grad(grad(g))), f, df)
        after = expand_derivatives(before)
        self.assertEqualTotalShape(before, after)
        # assert after == expected


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
@pytest.mark.parametrize("lower_alg", [True, False])
@pytest.mark.parametrize("lower_geo", [True, False])
@pytest.mark.parametrize("apply_deriv", [True, False])
def test_diff_grad_jacobian(cell, gdim, order, lower_alg, lower_geo, apply_deriv):
    tdim = cell.topological_dimension

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
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("lower_alg", [True, False])
@pytest.mark.parametrize("lower_geo", [True, False])
@pytest.mark.parametrize("apply_deriv", [True, False])
def test_diff_grad_grad_jacobian(cell, gdim, order, lower_alg, lower_geo, apply_deriv):
    tdim = cell.topological_dimension

    domain = Mesh(LagrangeElement(cell, order, (gdim,)))

    J = Jacobian(domain)
    assert J.ufl_shape == (gdim, tdim)

    F = grad(grad(J))

    if lower_alg:
        F = apply_algebra_lowering(F)

    if lower_geo:
        F = apply_geometry_lowering(F)

    if apply_deriv:
        F = apply_derivatives(F)

    assert F[:, :, :, :] != 0
    assert F.ufl_shape == (gdim, tdim, gdim, gdim)

    V = FunctionSpace(domain, LagrangeElement(cell, 1))
    u = Coefficient(V)

    δF_u = diff(F, u)

    if lower_alg:
        δF_u = apply_algebra_lowering(δF_u)

    if lower_geo:
        δF_u = apply_geometry_lowering(δF_u)

    δF_u = apply_derivatives(δF_u)

    assert δF_u == 0
    assert δF_u.ufl_shape == (gdim, tdim, gdim, gdim)
