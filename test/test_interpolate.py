"""Test Interpolate object."""

__authors__ = "Nacime Bouziani"
__date__ = "2021-11-19"

import pytest

from ufl import (Action, Adjoint, Argument, Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, action,
                 adjoint, derivative, dx, grad, inner, replace, triangle)
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.analysis import (extract_arguments, extract_arguments_and_coefficients, extract_base_form_operators,
                                     extract_coefficients)
from ufl.algorithms.expand_indices import expand_indices
from ufl.core.interpolate import Interpolate
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1


@pytest.fixture
def domain_2d():
    return Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))


@pytest.fixture
def V1(domain_2d):
    f1 = FiniteElement("CG", triangle, 1, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V2(domain_2d):
    f1 = FiniteElement("CG", triangle, 2, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


def test_symbolic(V1, V2):

    # Set dual of V2
    V2_dual = V2.dual()

    u = Coefficient(V1)
    vstar = Argument(V2_dual, 0)
    Iu = Interpolate(u, vstar)

    assert Iu == Interpolate(u, V2)
    assert Iu.ufl_function_space() == V2
    assert Iu.argument_slots() == (vstar, u)
    assert Iu.arguments() == (vstar,)
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
    assert extract_arguments_and_coefficients(Iu) == ([vstar], [u])

    F = Iu * dx
    # Form composition: Iu * dx <=> Action(v * dx, Iu(u; v*))
    assert extract_arguments(F) == []
    assert extract_arguments_and_coefficients(F) == ([], [u])

    for e in [Iu, F]:
        assert extract_coefficients(e) == [u]
        assert extract_base_form_operators(e) == [Iu]

    # -- Interpolate(u, V2) -- #
    Iv = Interpolate(uhat, V2)
    assert extract_arguments(Iv) == [vstar, uhat]
    assert extract_arguments_and_coefficients(Iv) == ([vstar, uhat], [])
    assert extract_coefficients(Iv) == []
    assert extract_base_form_operators(Iv) == [Iv]

    # -- Action(v * v2 * dx, Iv) -- #
    v2 = TrialFunction(V2)
    v = TestFunction(V1)
    F = Action(v * v2 * dx, Iv)
    assert extract_arguments(F) == [v, uhat]
