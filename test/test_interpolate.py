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
    exp,
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
from ufl.form import Form, FormSum
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


@pytest.fixture
def V3(domain_2d):
    f1 = FiniteElement("CG", triangle, 3, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V4(domain_2d):
    f1 = FiniteElement("CG", triangle, 4, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V5(domain_2d):
    f1 = FiniteElement("CG", triangle, 5, (), identity_pullback, H1)
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
    u = Argument(V1, 0)
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
    Iv = Interpolate(v1, vstar)  # V1 -> V2

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

    # action of one-form on interpolation operator
    one_form = Argument(V2, 0) * dx
    action_one_form = action(one_form, Iv)  # adjoint interpolation V2^* -> V1^*
    assert isinstance(action_one_form, Interpolate)
    assert action_one_form.arguments() == (Argument(V1, 0),)
    assert action_one_form.ufl_function_space() == V1.dual()

    # zero-form case
    action_zero_form = action(one_form, Iu)  # a number
    assert isinstance(action_zero_form, Form)
    assert action_zero_form.arguments() == ()


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

    # Derivative of form I(u, V2) wrt coefficient u
    J = Iu * dx
    dJdu = expand_derivatives(derivative(J, u))
    assert isinstance(dJdu, Interpolate)
    assert dJdu.arguments() == (Argument(V1, 0),)


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
    u0 = Argument(V1, 0)
    u1 = Argument(V1, 1)
    vstar0 = Argument(V2.dual(), 0)
    vstar1 = Argument(V2.dual(), 1)
    cofunc = Cofunction(V2.dual())
    one_form = Argument(V2, 0) * dx

    Interpolate(u, cofunc)
    Interpolate(u, vstar0)
    Interpolate(u, V2)
    Interpolate(u0, cofunc)  # adjoint
    Interpolate(u0, one_form)  # adjoint
    Interpolate(u1, vstar0)
    Interpolate(u0, vstar1)  # adjoint

    with pytest.raises(ValueError, match=r"Same argument numbers in first and second operands"):
        Interpolate(u0, vstar0)

    with pytest.raises(ValueError, match=r"Non-contiguous argument numbers in interpolate."):
        Interpolate(u, vstar1)

    with pytest.raises(ValueError, match=r"Non-contiguous argument numbers in interpolate."):
        Interpolate(u1, cofunc)

    with pytest.raises(ValueError, match=r"Non-contiguous argument numbers in interpolate."):
        Interpolate(u1, one_form)

    u2 = u0 * u1
    with pytest.raises(
        ValueError, match=r"Same argument numbers in first and second operands to interpolate."
    ):
        Interpolate(u2, vstar0)

    with pytest.raises(ValueError, match=r"Expecting a primal function space."):
        Interpolate(u, V2.dual())

    with pytest.raises(
        ValueError, match=r"Expecting the second argument to be FunctionSpace or BaseForm."
    ):
        Interpolate(u, u0)


def test_interpolate_composition(V1, V2, V3, V4, V5):
    u5 = Coefficient(V5)
    u4 = Interpolate(u5, V4)
    assert u4.arguments() == (Argument(V4.dual(), 0),)
    u3 = Interpolate(u4, V3)
    assert u3.arguments() == (Argument(V3.dual(), 0),)
    u2 = Interpolate(u3, V2)
    u1 = Interpolate(u2, V1)

    assert u4.ufl_function_space() == V4
    assert u3.ufl_function_space() == V3
    assert u2.ufl_function_space() == V2
    assert u1.ufl_function_space() == V1

    args = extract_arguments(u1)
    assert set(args) == {
        Argument(V4.dual(), 0),
        Argument(V3.dual(), 0),
        Argument(V2.dual(), 0),
        Argument(V1.dual(), 0),
    }

    assert u1.arguments() == (Argument(V1.dual(), 0),)

    # adjoint
    u1 = Cofunction(V1.dual())
    u2 = Interpolate(Argument(V2, 0), u1)  # V1^* x V2 -> R, equiv V1^* -> V2^*
    u3 = Interpolate(Argument(V3, 0), u2)
    u4 = Interpolate(Argument(V4, 0), u3)
    u5 = Interpolate(Argument(V5, 0), u4)

    assert u2.ufl_function_space() == V2.dual()
    assert u3.ufl_function_space() == V3.dual()
    assert u4.ufl_function_space() == V4.dual()
    assert u5.ufl_function_space() == V5.dual()

    args = extract_arguments(u5)
    assert set(args) == {
        Argument(V5, 0),
        Argument(V4, 0),
        Argument(V3, 0),
        Argument(V2, 0),
        Argument(V1, 0),
    }
    assert u5.arguments() == (Argument(V5, 0),)

    forward_twoform = Interpolate(Argument(V2, 1), V1)  # V2 x V1^* -> R, equiv V2 -> V1
    assert forward_twoform.ufl_function_space() == V1
    assert forward_twoform.arguments() == (Argument(V1.dual(), 0), Argument(V2, 1))
    Iu = Interpolate(forward_twoform, V3)  # V2 x V3^* -> R, equiv V2 -> V3
    assert Iu.ufl_function_space() == V3
    assert Iu.arguments() == (Argument(V3.dual(), 0), Argument(V2, 1))
    assert set(extract_arguments(Iu)) == {
        Argument(V2, 1),
        Argument(V3.dual(), 0),
        Argument(V1.dual(), 0),
    }

    adjoint_twoform = Interpolate(
        Argument(V2, 0), Argument(V1.dual(), 1)
    )  # V1^* x V2 -> R, equiv V1^* -> V2^*
    assert adjoint_twoform.ufl_function_space() == V2.dual()
    assert adjoint_twoform.arguments() == (Argument(V2, 0), Argument(V1.dual(), 1))
    Iu_adj = Interpolate(Argument(V3, 0), adjoint_twoform)  # V1^* x V3 -> R, equiv V1^* -> V3^*
    assert Iu_adj.ufl_function_space() == V3.dual()
    assert Iu_adj.arguments() == (Argument(V3, 0), Argument(V1.dual(), 1))
    assert set(extract_arguments(Iu_adj)) == {
        Argument(V3, 0),
        Argument(V2, 0),
        Argument(V1.dual(), 1),
    }

    with pytest.raises(ValueError, match=r"Expecting the first argument to be primal."):
        Interpolate(u2, V3)


def test_interpolate_expr(V1, V2, V3, V4):
    u4 = Coefficient(V4)
    u31 = Interpolate(u4, V3)
    u32 = Interpolate(u4, V3)
    u3 = u31 + u32
    u2 = Interpolate(u3, V2)
    u1 = Interpolate(u2, V1)

    assert extract_arguments(u3) == []
    assert u1.ufl_function_space() == V1
    assert set(extract_arguments(u1)) == {
        Argument(V3.dual(), 0),
        Argument(V2.dual(), 0),
        Argument(V1.dual(), 0),
    }
    assert u1.arguments() == (Argument(V1.dual(), 0),)
    assert u2.arguments() == (Argument(V2.dual(), 0),)

    # product
    u33 = u31 * u32
    assert extract_arguments(u33) == []
    u21 = Interpolate(u33, V2)
    u11 = Interpolate(u21, V1)
    assert u11.arguments() == (Argument(V1.dual(), 0),)
    assert u21.arguments() == (Argument(V2.dual(), 0),)

    # with MathFunction
    u34 = exp(u31 * u32) + u31 - 1
    assert extract_arguments(u34) == []
    u22 = Interpolate(u34, V2)
    u12 = Interpolate(u22, V1)
    assert u12.arguments() == (Argument(V1.dual(), 0),)
    assert u22.arguments() == (Argument(V2.dual(), 0),)

    # adjoint
    u1 = Cofunction(V1.dual())
    u21 = Interpolate(Argument(V2, 0), u1)
    u22 = Interpolate(Argument(V2, 0), u1)
    u2 = u21 + u22
    u3 = Interpolate(Argument(V3, 0), u2)
    u4 = Interpolate(Argument(V4, 0), u3)

    assert u1.arguments() == (Argument(V1, 0),)
    assert extract_arguments(u2) == [Argument(V2, 0)]
    assert u4.ufl_function_space() == V4.dual()
    assert set(extract_arguments(u4)) == {Argument(V2, 0), Argument(V3, 0), Argument(V4, 0)}
    assert u4.arguments() == (Argument(V4, 0),)
    assert u3.arguments() == (Argument(V3, 0),)


def test_interpolate_cofunction(V1, V2):
    V1_cofunc = Cofunction(V1.dual())
    V2_test = TestFunction(V2)

    Iu_adj = Interpolate(V2_test, V1_cofunc)  # Cofunction in V2^*
    assert Iu_adj.arguments() == (Argument(V2, 0),)

    # Test zero-form
    V1_coeff = Coefficient(V1)
    V2_coeff = Interpolate(V1_coeff, V2)
    Iu_zeroform = Interpolate(V2_coeff, V1_cofunc)
    assert Iu_zeroform.arguments() == ()

    Iu2 = Interpolate(Iu_zeroform * V1_coeff, V2)
    assert Iu2.arguments() == (Argument(V2.dual(), 0),)

    Iu3 = V1_coeff * Iu_zeroform
    assert extract_arguments(Iu3) == []


def test_interpolate_form(V1, V2):
    V1_coeff = Coefficient(V1)
    V1_test = TestFunction(V1)
    V1_trial = TrialFunction(V1)

    Iu = Interpolate(V1_coeff, V2)
    F = V1_test * Iu * dx

    assert extract_arguments(Iu) == [Argument(V2.dual(), 0)]
    assert extract_arguments(F) == [V1_test]
    assert extract_terminals_with_domain(F) == ([V1_test], [V1_coeff], [])

    # Forward interpolation moves trial function from V1 to V2
    V2_trial = TrialFunction(V2)
    Iu = Interpolate(V2_trial, V1)  # V2 x V1^* -> R, equiv V2 -> V1
    F = Iu * V1_test * dx
    assert extract_arguments(Iu) == [Argument(V1.dual(), 0), V2_trial]
    assert extract_arguments(F) == [V1_test, V2_trial]
    assert F.arguments() == (V1_test, V2_trial)

    F_act = V1_trial * V1_test * dx  # V1 x V1 -> R, equiv V1 -> V1^*
    assert F_act.arguments() == (V1_test, V1_trial)
    G = action(F_act, Iu)  # V2 -> V1 -> V1^*, equiv V2 x V1 -> R
    assert extract_arguments(G) == [V1_test, V2_trial]
    assert G.arguments() == (V1_test, V2_trial)

    # Adjoint interpolation moves test function from V1 to V2
    V2_test = TestFunction(V2)
    Iu_adj = Interpolate(V2_test, V1)  # V1^* x V2 -> R, equiv V1^* -> V2^*
    assert extract_arguments(Iu_adj) == [Argument(V2, 0), Argument(V1.dual(), 1)]
    assert Iu_adj.arguments() == (Argument(V2, 0), Argument(V1.dual(), 1))
    G2 = action(Iu_adj, F_act)  # V1 -> V1^* -> V2^*, equiv V1 x V2 -> R
    assert extract_arguments(G2) == [V2_test, V1_trial]
    assert G2.arguments() == (V2_test, V1_trial)

    # alternatively using adjoint(Iu)
    Iu_adj = adjoint(Iu)
    G2 = action(Iu_adj, F_act)
    assert extract_arguments(G2) == [V2_test, V1_trial]
    assert G2.arguments() == (V2_test, V1_trial)

    # Adjoint interpolation of Form
    G3 = Interpolate(V2_test, F_act)
    assert G3.arguments() == (V2_test, V1_trial)
    assert G3.ufl_function_space() == V2.dual()


def test_interpolate_adjoint(V1, V2):
    V1_trial = TrialFunction(V1)
    Iu_forward = Interpolate(V1_trial, V2)  # V1 x V2^* -> R, equiv V1 -> V2
    Iu_adjoint = adjoint(Iu_forward)  # V2^* x V1 -> R, equiv V2^* -> V1^*

    assert Iu_forward.arguments() == (Argument(V2.dual(), 0), V1_trial)
    assert Iu_forward.ufl_function_space() == V2
    assert Iu_adjoint.arguments() == (Argument(V1, 0), Argument(V2.dual(), 1))

    Iu_adjoint2 = Interpolate(Argument(V1, 0), Argument(V2.dual(), 1))
    assert Iu_adjoint2.arguments() == Iu_adjoint.arguments()

    Iu = adjoint(Iu_adjoint)
    assert Iu.arguments() == Iu_forward.arguments()


def test_interpolate_formsum(V1, V2):
    V1_cofunc = Cofunction(V1.dual())
    V1_cofunc2 = Cofunction(V1.dual())
    V1_sum = V1_cofunc + V1_cofunc2
    assert isinstance(V1_sum, FormSum)
    assert V1_sum.arguments() == (Argument(V1, 0),)

    V2_cofunc = Interpolate(Argument(V2, 0), V1_sum)  # Cofunction in V2^*
    assert V2_cofunc.arguments() == (Argument(V2, 0),)
    assert V2_cofunc.ufl_function_space() == V2.dual()
