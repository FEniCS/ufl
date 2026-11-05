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
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    action,
    conditional,
    derivative,
    dS,
    ds,
    dx,
    exp,
    interval,
    lhs,
    rhs,
    system,
)


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
    for actual_a, actual_L in [(lhs(F), rhs(F)), (a, L)]:
        for coordinate, u_weight, f_weight in [(1, ut, ft), (-1, uf, ff)]:
            mapping = {u: 2, v: 3, f: 5}
            a_value = sum(i.integrand()((coordinate,), mapping) for i in actual_a.integrals())
            L_value = sum(i.integrand()((coordinate,), mapping) for i in actual_L.integrals())
            assert a_value == 6 * u_weight
            assert L_value == 15 * f_weight


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
