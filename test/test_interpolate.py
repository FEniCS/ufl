"""Test Interpolate object."""

__authors__ = "Nacime Bouziani"
__date__ = "2021-11-19"

import pytest
from utils import FiniteElement, LagrangeElement

from ufl import (
    Action,
    Adjoint,
    Argument,
    Coefficient,
    Cofunction,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    action,
    adjoint,
    derivative,
    dx,
    grad,
    inner,
    replace,
    triangle,
)
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.analysis import (
    extract_arguments,
    extract_base_form_operators,
    extract_coefficients,
    extract_terminals_with_domain,
)
from ufl.algorithms.expand_indices import expand_indices
from ufl.core.interpolate import Interpolate
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1


@pytest.fixture
def domain_2d():
    return Mesh(LagrangeElement(triangle, 1, (2,)))


@pytest.fixture
def V1(domain_2d):
    f1 = FiniteElement("CG", triangle, 1, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V2(domain_2d):
    f1 = FiniteElement("CG", triangle, 2, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


def test_symbolic(V1, V2):
    u = Coefficient(V1)
    vstar = Argument(V2.dual(), 0)
    Iu = Interpolate(u, vstar)

    assert Iu == Interpolate(u, V2)
    assert Iu.ufl_function_space() == V2
    assert Iu.argument_slots() == (vstar, u)
    assert Iu.arguments() == (vstar,)
    assert Iu.ufl_operands == (u,)


def test_symbolic_adjoint(V1, V2):
    u = Argument(V1, 1)
    form = inner(1, Argument(V2, 0)) * dx
    cofun = Cofunction(V2.dual())

    for vstar in (form, cofun):
        Iu = Interpolate(u, vstar)

        assert Iu.ufl_function_space() == V1.dual()
        assert Iu.argument_slots() == (vstar, u)
        assert Iu.arguments() == (u,)
        assert Iu.ufl_operands == (u,)


def test_action_adjoint(V1, V2):
    # Set dual of V2
    V2_dual = V2.dual()
    vstar = Argument(V2_dual, 0)

    u = Coefficient(V1)
    Iu = Interpolate(u, vstar)

    v1 = TrialFunction(V1)
    Iv = Interpolate(v1, vstar)

    assert Iv.argument_slots() == (vstar, v1)
    assert Iv.arguments() == (vstar, v1)

    # -- Action -- #
    v = TestFunction(V1)
    v2 = TrialFunction(V2)
    F = v2 * v * dx
    assert action(Iv, u) == Action(Iv, u)
    assert action(F, Iv) == Action(F, Iv)
    assert action(F, Iu) == Iu * v * dx

    # -- Adjoint -- #
    adjoint(Iv) == Adjoint(Iv)


def test_differentiation(V1, V2):
    u = Coefficient(V1)
    v = TestFunction(V1)

    # Define Interpolate
    Iu = Interpolate(u, V2)

    # -- Differentiate: Interpolate(u, V2) -- #
    uhat = TrialFunction(V1)
    dIu = expand_derivatives(derivative(Iu, u, uhat))

    # dInterpolate(u, v*)/du[uhat] <==> Interpolate(uhat, v*)
    assert dIu == Interpolate(uhat, V2)

    # -- Differentiate: Interpolate(u**2, V2) -- #
    g = u**2
    Ig = Interpolate(g, V2)
    dIg = expand_derivatives(derivative(Ig, u, uhat))
    assert dIg == Interpolate(2 * uhat * u, V2)

    # -- Differentiate: I(u, V2) * v * dx -- #
    F = Iu * v * dx
    Ihat = TrialFunction(Iu.ufl_function_space())
    dFdu = expand_derivatives(derivative(F, u, uhat))
    # Compute dFdu = ∂F/∂u + Action(dFdIu, dIu/du)
    #              = Action(dFdIu, Iu(uhat, v*))
    dFdIu = expand_derivatives(derivative(F, Iu, Ihat))
    assert dFdIu == Ihat * v * dx
    assert dFdu == Action(dFdIu, dIu)

    # -- Differentiate: u * I(u, V2) * v * dx -- #
    F = u * Iu * v * dx
    dFdu = expand_derivatives(derivative(F, u, uhat))
    # Compute dFdu = ∂F/∂u + Action(dFdIu, dIu/du)
    #              = ∂F/∂u + Action(dFdIu, Iu(uhat, v*))
    dFdu_partial = uhat * Iu * v * dx
    dFdIu = Ihat * u * v * dx
    assert dFdu == dFdu_partial + Action(dFdIu, dIu)

    # -- Differentiate (wrt Iu): <Iu, v> + <grad(Iu), grad(v)> - <f, v>
    f = Coefficient(V1)
    F = inner(Iu, v) * dx + inner(grad(Iu), grad(v)) * dx - inner(f, v) * dx
    dFdIu = expand_derivatives(derivative(F, Iu, Ihat))

    # BaseFormOperators are treated as coefficients when a form is differentiated wrt them.
    # -> dFdIu <=> dFdw
    w = Coefficient(V2)
    F = replace(F, {Iu: w})
    dFdw = expand_derivatives(derivative(F, w, Ihat))

    # Need to expand indices to be able to match equal (different MultiIndex used for both).
    assert expand_indices(dFdIu) == expand_indices(dFdw)


def test_extract_base_form_operators(V1, V2):
    u = Coefficient(V1)
    uhat = TrialFunction(V1)
    vstar = Argument(V2.dual(), 0)

    # -- Interpolate(u, V2) -- #
    Iu = Interpolate(u, V2)
    assert extract_arguments(Iu) == [vstar]
    assert extract_terminals_with_domain(Iu) == ([vstar], [u], [])

    F = Iu * dx
    # Form composition: Iu * dx <=> Action(v * dx, Iu(u; v*))
    assert extract_arguments(F) == []
    assert extract_terminals_with_domain(F) == ([], [u], [])

    for e in [Iu, F]:
        assert extract_coefficients(e) == [u]
        assert extract_base_form_operators(e) == [Iu]

    # -- Interpolate(u, V2) -- #
    Iv = Interpolate(uhat, V2)
    assert extract_arguments(Iv) == [vstar, uhat]
    assert extract_terminals_with_domain(Iv) == ([vstar, uhat], [], [])
    assert extract_coefficients(Iv) == []
    assert extract_base_form_operators(Iv) == [Iv]

    # -- Action(v * v2 * dx, Iv) -- #
    v2 = TrialFunction(V2)
    v = TestFunction(V1)
    F = Action(v * v2 * dx, Iv)
    assert extract_arguments(F) == [v, uhat]


def test_operator_derivative_reconstruction(V1, V2):
    u = Coefficient(V1)

    # Define Interpolate
    Iu = Interpolate(u, V2)

    # -- Differentiate: Interpolate(u, V2) -- #
    uhat = TrialFunction(V1)
    dIu = derivative(Iu, u, uhat)
    # Check operator derivative can be reconstructed
    dIu_r = dIu._ufl_expr_reconstruct_(*dIu.ufl_operands)
    dIu = expand_derivatives(dIu_r)

    assert dIu == Interpolate(uhat, V2)


def test_interpolate_argument_numbering(V1, V2):
    u = Coefficient(V1)
    vstar = Argument(V2.dual(), 0)

    # Valid numbering
    Interpolate(u, vstar)
    Interpolate(u, V2)

    u0 = Argument(V1, 0)
    u1 = Argument(V1, 1)

    # Invalid numbering
    with pytest.raises(ValueError, match="Same argument numbers in first and second operands"):
        Interpolate(u0, vstar)

    vstar1 = Argument(V2.dual(), 1)
    with pytest.raises(ValueError, match="Non-contiguous argument numbers in interpolate."):
        Interpolate(u, vstar1)

    cofunc = Cofunction(V2.dual())
    with pytest.raises(ValueError, match="Non-contiguous argument numbers in interpolate."):
        Interpolate(u1, cofunc)

    u2 = u0 * u1
    with pytest.raises(ValueError, match="Can only interpolate expressions with zero or one argument."):
        Interpolate(u2, vstar)
