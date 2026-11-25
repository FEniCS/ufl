__authors__ = "Marie E. Rognes"

# First added: 2011-11-09
# Last changed: 2011-11-09

import pytest
from utils import LagrangeElement

from ufl import (
    Argument,
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    MixedFunctionSpace,
    SpatialCoordinate,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    action,
    conditional,
    derivative,
    dS,
    ds,
    dx,
    exp,
    extract_blocks,
    grad,
    indices,
    inner,
    interval,
    lhs,
    rhs,
    system,
    triangle,
)
from ufl.form import ZeroBaseForm
from ufl.algorithms import compute_form_data
from ufl.algorithms.formtransformations import compute_form_arities


def test_lhs_rhs_simple():
    V = LagrangeElement(interval, 1)
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, V)
    v = TestFunction(space)
    u = TrialFunction(space)
    w = Argument(space, 2)  # This was 0, not sure why
    f = Coefficient(space)

    F0 = f * u * v * w * dx
    a, L = system(F0)
    assert len(a.integrals()) == 0
    assert len(L.integrals()) == 0

    F1 = derivative(F0, f)
    a, L = system(F1)
    assert len(a.integrals()) == 0
    assert len(L.integrals()) == 0

    F2 = action(F0, f)
    a, L = system(F2)
    assert len(a.integrals()) == 1
    assert len(L.integrals()) == 0

    F3 = action(F2, f)
    a, L = system(F3)
    assert len(L.integrals()) == 1


@pytest.mark.parametrize(
    "weights", [(1, 1, 2, 2), (0, 0, 2, 2), (1, 1, 0, 0), (1, 1, 0, 2), (0, 1, 2, 2)]
)
def test_lhs_rhs_conditional(weights):
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, LagrangeElement(interval, 1))
    u, v, f = TrialFunction(space), TestFunction(space), Coefficient(space)
    x = SpatialCoordinate(domain)
    ut, ft, uf, ff = weights
    F = conditional(x[0] > 0, ut * u - ft * f, uf * u - ff * f) * v * dx

    a, L = system(F)
    assert a == conditional(x[0] > 0, ut * u, uf * u) * v * dx
    # rhs negates outside the conditional, so compare L by value
    mapping = {u: 2, v: 3, f: 5}
    for coordinate, f_weight in [(1, ft), (-1, ff)]:
        assert sum(i.integrand()((coordinate,), mapping) for i in L.integrals()) == 15 * f_weight


def test_lhs_rhs_conditional_different_arguments_raises():
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, LagrangeElement(interval, 1))
    v, u, w = TestFunction(space), TrialFunction(space), Argument(space, 2)
    x = SpatialCoordinate(domain)

    with pytest.raises(ValueError, match="conditionals with different Arguments"):
        compute_form_arities(conditional(x[0] > 0, u, w) * v * dx)


def test_lhs_rhs_conditional_condition_with_argument_raises():
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, LagrangeElement(interval, 1))
    u, v, f = TrialFunction(space), TestFunction(space), Coefficient(space)

    with pytest.raises(ValueError, match="Condition cannot depend on form arguments"):
        system(conditional(u > 0, f, 2 * f) * v * dx)


@pytest.mark.parametrize("kind", ["vector", "free_index", "restricted"])
def test_lhs_rhs_conditional_splits(kind):
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, LagrangeElement(interval, 1))
    u, v, f = TrialFunction(space), TestFunction(space), Coefficient(space)
    x = SpatialCoordinate(domain)
    c = x[0] > 0
    (i,) = indices(1)
    F = {
        # The zero branch must match the shape and free indices of the conditional
        "vector": inner(conditional(c, grad(u), grad(f)), grad(v)) * dx,
        "free_index": conditional(c, u.dx(i), f.dx(i)) * v.dx(i) * dx,
        "restricted": conditional(x[0]("+") > 0, u("+") - f("+"), 0) * v("+") * dS,
    }[kind]

    a, L = system(F)
    assert compute_form_arities(F) == {1, 2}
    assert compute_form_arities(a) == {2}
    assert compute_form_arities(L) == {1}
    compute_form_data(a)
    compute_form_data(L)


@pytest.mark.parametrize("inner_branch", ["true", "false"])
def test_lhs_rhs_conditional_nested(inner_branch):
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, LagrangeElement(interval, 1))
    u, v, f = TrialFunction(space), TestFunction(space), Coefficient(space)
    x = SpatialCoordinate(domain)
    c, d = x[0] > 0, x[0] > 0.5
    if inner_branch == "true":
        F = conditional(c, conditional(d, u - f, 3 * u), 2 * f) * v * dx
        expected_a = conditional(c, conditional(d, u, 3 * u), 0) * v * dx
    else:
        F = conditional(c, u - f, conditional(d, 2 * f, 4 * u)) * v * dx
        expected_a = conditional(c, u, conditional(d, 0, 4 * u)) * v * dx

    a, L = system(F)
    assert a == expected_a
    mapping = {u: 2, v: 3, f: 5}
    for coordinate in [(1,), (0.25,), (-1,)]:
        F_value, a_value, L_value = (
            sum(i.integrand()(coordinate, mapping) for i in form.integrals()) for form in (F, a, L)
        )
        assert a_value - L_value == F_value


def test_lhs_rhs_conditional_derivative_term():
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, LagrangeElement(interval, 2))
    u, v, f = TrialFunction(space), TestFunction(space), Coefficient(space)
    c = SpatialCoordinate(domain)[0] > 0

    # Both rank-one terms of a branch are kept, the rank-zero term goes to L
    a, L = system(conditional(c, u + 2 * grad(u)[0] - f, 3 * u - 2 * f) * v * dx)
    assert a == conditional(c, u + 2 * grad(u)[0], 3 * u) * v * dx
    mapping = {v: 3, f: 5}
    for coordinate, L_value in [(1, 15), (-1, 30)]:
        assert sum(i.integrand()((coordinate,), mapping) for i in L.integrals()) == L_value


@pytest.mark.parametrize("branches", ["velocity_pressure_and_velocity", "velocity_and_pressure"])
def test_lhs_rhs_conditional_mixed_function_space(branches):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 2, (2,)))
    Q = FunctionSpace(domain, LagrangeElement(triangle, 1))
    W = MixedFunctionSpace(V, Q)
    u, p = TrialFunctions(W)
    v, _ = TestFunctions(W)
    f = Coefficient(Q)
    c = SpatialCoordinate(domain)[0] > 0
    if branches == "velocity_pressure_and_velocity":
        F = conditional(c, u[0] + p - f, 2 * u[0] - f) * v[0] * dx
        expected_uu = conditional(c, u[0], 2 * u[0]) * v[0] * dx
        expected_up = conditional(c, p, 0) * v[0] * dx
    else:
        F = conditional(c, u[0] - f, p) * v[0] * dx
        expected_uu = conditional(c, u[0], 0) * v[0] * dx
        expected_up = conditional(c, 0, p) * v[0] * dx

    # A branch without the trial function of a block is zero in that block
    a, L = system(F)
    (a_uu, a_up), (a_pu, a_pp) = extract_blocks(a)
    assert a_uu == expected_uu
    assert a_up == expected_up
    assert a_pu is None and a_pp is None

    mapping = {u: (2, 4), p: 7, v: (3, 1), f: 5}
    for coordinate in [(1, 0), (-1, 0)]:
        F_value, a_value, L_value = (
            sum(i.integrand()(coordinate, mapping) for i in form.integrals()) for form in (F, a, L)
        )
        assert a_value - L_value == F_value


def test_lhs_rhs_derivatives():
    V = LagrangeElement(interval, 1)
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, V)
    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)

    F0 = exp(f) * u * v * dx + v * dx + f * v * ds + exp(f)("+") * v * dS
    a, L = system(F0)
    assert len(a.integrals()) == 1
    assert len(L.integrals()) == 3

    derivative(F0, f)
    a, L = system(F0)


def test_lhs_rhs_zero_base_form():
    V = LagrangeElement(interval, 1)
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, V)
    v = TestFunction(space)
    u = TrialFunction(space)

    zero = ZeroBaseForm((v, u))
    assert lhs(zero) == zero
    assert rhs(zero) == ZeroBaseForm((v,))
    a, L = system(zero)
    assert a == zero
    assert L == ZeroBaseForm((v,))

    # A form whose derivative vanishes expands to a ZeroBaseForm inside lhs and rhs
    f = Coefficient(space)
    F = v * dx
    assert lhs(derivative(F, f, u)) == zero
    assert rhs(derivative(F, f, u)) == ZeroBaseForm((v,))


def test_lhs_rhs_slightly_obscure():
    V = LagrangeElement(interval, 1)
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    space = FunctionSpace(domain, V)
    u = TrialFunction(space)
    w = Argument(space, 2)
    f = Constant(domain)

    # FIXME:
    # ufl.algorithsm.formtransformations.compute_form_with_arity
    # is not perfect, e.g. try
    # F = f*u*w*dx + f*w*dx
    F = f * u * w * dx
    a, L = system(F)
    assert len(a.integrals()) == 1
    assert len(L.integrals()) == 0

    F = f * w * dx
    a, L = system(F)
    assert len(L.integrals()) == 1
