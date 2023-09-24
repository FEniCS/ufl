"""Test ExternalOperator object."""

__authors__ = "Nacime Bouziani"
__date__ = "2019-03-26"

import pytest

from ufl import (Action, Argument, Coefficient, Constant, Form, FunctionSpace, Mesh, TestFunction, TrialFunction,
                 action, adjoint, cos, derivative, dx, inner, sin, triangle)
from ufl.algorithms import expand_derivatives
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.core.external_operator import ExternalOperator
from ufl.finiteelement import FiniteElement
from ufl.form import BaseForm
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1


@pytest.fixture
def domain_2d():
    return Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))


@pytest.fixture
def V1(domain_2d):
    f1 = FiniteElement("CG", triangle, 1, (), identity_pull_back, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V2(domain_2d):
    f1 = FiniteElement("CG", triangle, 2, (), identity_pull_back, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V3(domain_2d):
    f1 = FiniteElement("CG", triangle, 3, (), identity_pull_back, H1)
    return FunctionSpace(domain_2d, f1)


def test_properties(V1):
    u = Coefficient(V1, count=0)
    r = Coefficient(V1, count=1)

    e = ExternalOperator(u, r, function_space=V1)

    assert e.ufl_function_space() == V1
    assert e.ufl_operands[0] == u
    assert e.ufl_operands[1] == r
    assert e.derivatives == (0, 0)
    assert e.ufl_shape == ()

    e2 = ExternalOperator(u, r, function_space=V1, derivatives=(3, 4))
    assert e2.derivatives == (3, 4)
    assert e2.ufl_shape == ()

    # Test __str__
    s = Coefficient(V1, count=2)
    t = Coefficient(V1, count=3)
    v0 = Argument(V1, 0)
    v1 = Argument(V1, 1)

    e = ExternalOperator(u, function_space=V1)
    assert str(e) == 'e(w_0; v_0)'

    e = ExternalOperator(u, function_space=V1, derivatives=(1,))
    assert str(e) == '∂e(w_0; v_0)/∂o1'

    e = ExternalOperator(u, r, 2 * s, t, function_space=V1, derivatives=(1, 0, 1, 2), argument_slots=(v0, v1))
    assert str(e) == '∂e(w_0, w_1, 2 * w_2, w_3; v_1, v_0)/∂o1∂o3∂o4∂o4'


def test_form(V1, V2):
    u = Coefficient(V1)
    m = Coefficient(V1)
    u_hat = TrialFunction(V1)
    v = TestFunction(V1)

    # F = N * v * dx
    N = ExternalOperator(u, m, function_space=V2)
    F = N * v * dx
    actual = derivative(F, u, u_hat)

    vstar, = N.arguments()
    Nhat = TrialFunction(N.ufl_function_space())

    dNdu = N._ufl_expr_reconstruct_(u, m, derivatives=(1, 0), argument_slots=(vstar, u_hat))
    dFdN = Nhat * v * dx
    expected = Action(dFdN, dNdu)

    assert apply_derivatives(actual) == expected

    # F = N * u * v * dx
    N = ExternalOperator(u, m, function_space=V1)
    F = N * u * v * dx
    actual = derivative(F, u, u_hat)

    vstar, = N.arguments()
    Nhat = TrialFunction(N.ufl_function_space())

    dNdu = N._ufl_expr_reconstruct_(u, m, derivatives=(1, 0), argument_slots=(vstar, u_hat))
    dFdu_partial = N * u_hat * v * dx
    dFdN = Nhat * u * v * dx
    expected = dFdu_partial + Action(dFdN, dNdu)
    assert apply_derivatives(actual) == expected


def test_differentiation_procedure_action(V1, V2):
    s = Coefficient(V1)
    u = Coefficient(V2)
    m = Coefficient(V2)

    # External operators
    N1 = ExternalOperator(u, m, function_space=V1)
    N2 = ExternalOperator(cos(s), function_space=V1)

    # Check arguments and argument slots
    assert len(N1.arguments()) == 1
    assert len(N2.arguments()) == 1
    assert N1.arguments() == N1.argument_slots()
    assert N2.arguments() == N2.argument_slots()

    # Check coefficients
    assert N1.coefficients() == (u, m)
    assert N2.coefficients() == (s,)

    # Get v*
    vstar_N1, = N1.arguments()
    vstar_N2, = N2.arguments()
    assert vstar_N1.ufl_function_space().dual() == V1
    assert vstar_N2.ufl_function_space().dual() == V1

    u_hat = Argument(V1, 1)
    s_hat = Argument(V2, 1)
    w = Coefficient(V1)
    r = Coefficient(V2)

    # Bilinear forms
    a1 = inner(N1, m) * dx
    Ja1 = derivative(a1, u, u_hat)
    Ja1 = expand_derivatives(Ja1)

    a2 = inner(N2, m) * dx
    Ja2 = derivative(a2, s, s_hat)
    Ja2 = expand_derivatives(Ja2)

    # Get external operators
    assert isinstance(Ja1, Action)
    dN1du = Ja1.right()
    dN1du_action = Action(dN1du, w)

    assert isinstance(Ja2, Action)
    dN2du = Ja2.right()
    dN2du_action = Action(dN2du, r)

    # Check shape
    assert dN1du.ufl_shape == ()
    assert dN2du.ufl_shape == ()

    # Get v*s
    vstar_dN1du, _ = dN1du.arguments()
    vstar_dN2du, _ = dN2du.arguments()
    assert vstar_dN1du.ufl_function_space().dual() == V1  # shape: (2,)
    assert vstar_dN2du.ufl_function_space().dual() == V1  # shape: (2,)

    # Check derivatives
    assert dN1du.derivatives == (1, 0)
    assert dN2du.derivatives == (1,)

    # Check arguments
    assert dN1du.arguments() == (vstar_dN1du, u_hat)
    assert dN1du_action.arguments() == (vstar_dN1du,)

    assert dN2du.arguments() == (vstar_dN2du, s_hat)
    assert dN2du_action.arguments() == (vstar_dN2du,)

    # Check argument slots
    assert dN1du.argument_slots() == (vstar_dN1du, u_hat)
    assert dN2du.argument_slots() == (vstar_dN2du, - sin(s) * s_hat)


def test_extractions(domain_2d, V1):
    from ufl.algorithms.analysis import (extract_arguments, extract_arguments_and_coefficients,
                                         extract_base_form_operators, extract_coefficients, extract_constants)

    u = Coefficient(V1)
    c = Constant(domain_2d)

    e = ExternalOperator(u, c, function_space=V1)
    vstar_e, = e.arguments()

    assert extract_coefficients(e) == [u]
    assert extract_arguments(e) == [vstar_e]
    assert extract_arguments_and_coefficients(e) == ([vstar_e], [u])
    assert extract_constants(e) == [c]
    assert extract_base_form_operators(e) == [e]

    F = e * dx

    assert extract_coefficients(F) == [u]
    assert extract_arguments(e) == [vstar_e]
    assert extract_arguments_and_coefficients(e) == ([vstar_e], [u])
    assert extract_constants(F) == [c]
    assert F.base_form_operators() == (e,)

    u_hat = Argument(V1, 1)
    e = ExternalOperator(u, function_space=V1, derivatives=(1,), argument_slots=(vstar_e, u_hat))

    assert extract_coefficients(e) == [u]
    assert extract_arguments(e) == [vstar_e, u_hat]
    assert extract_arguments_and_coefficients(e) == ([vstar_e, u_hat], [u])
    assert extract_base_form_operators(e) == [e]

    F = e * dx

    assert extract_coefficients(F) == [u]
    assert extract_arguments(e) == [vstar_e, u_hat]
    assert extract_arguments_and_coefficients(e) == ([vstar_e, u_hat], [u])
    assert F.base_form_operators() == (e,)

    w = Coefficient(V1)
    e2 = ExternalOperator(w, e, function_space=V1)
    vstar_e2, = e2.arguments()

    assert extract_coefficients(e2) == [u, w]
    assert extract_arguments(e2) == [vstar_e2, u_hat]
    assert extract_arguments_and_coefficients(e2) == ([vstar_e2, u_hat], [u, w])
    assert extract_base_form_operators(e2) == [e, e2]

    F = e2 * dx

    assert extract_coefficients(e2) == [u, w]
    assert extract_arguments(e2) == [vstar_e2, u_hat]
    assert extract_arguments_and_coefficients(e2) == ([vstar_e2, u_hat], [u, w])
    assert F.base_form_operators() == (e, e2)


def get_external_operators(form_base):
    if isinstance(form_base, ExternalOperator):
        return (form_base,)
    elif isinstance(form_base, BaseForm):
        return form_base.base_form_operators()
    else:
        raise ValueError('Expecting FormBase argument!')


def test_adjoint_action_jacobian(V1, V2, V3):

    u = Coefficient(V1)
    m = Coefficient(V2)

    # N(u, m; v*)
    N = ExternalOperator(u, m, function_space=V3)
    vstar_N, = N.arguments()

    # Arguments for the Gateaux-derivative
    def u_hat(number):
        return Argument(V1, number)   # V1: degree 1 # dFdu.arguments()[-1]

    def m_hat(number):
        return Argument(V2, number)   # V2: degree 2 # dFdm.arguments()[-1]

    def vstar_N(number):
        return Argument(V3.dual(), number)  # V3: degree 3

    # Coefficients for the action
    w = Coefficient(V1)  # for u
    p = Coefficient(V2)  # for m

    v2 = TestFunction(V2)
    v3 = TestFunction(V3)
    form_base_expressions = (N * dx, N * v2 * dx, N * v3 * dx)  # , N)

    for F in form_base_expressions:

        # Get test function
        v_F = F.arguments() if isinstance(F, Form) else ()
        # If we have a 0-form with an ExternalOperator: e.g. F = N * dx
        # => F.arguments() = (), because of form composition.
        # But we still need to make arguments with number 1 (i.e. n_arg = 1)
        # since at the external operator level, argument numbering is based on
        # the external operator arguments and not on the outer form arguments.
        n_arg = len(v_F) if len(v_F) else 1
        assert n_arg < 2

        # Differentiate
        dFdu = expand_derivatives(derivative(F, u, u_hat(n_arg)))
        dFdm = expand_derivatives(derivative(F, m, m_hat(n_arg)))

        assert dFdu.arguments() == v_F + (u_hat(n_arg),)
        assert dFdm.arguments() == v_F + (m_hat(n_arg),)

        assert isinstance(dFdu, Action)

        # dNdu(u, m; u_hat, v*)
        dNdu = dFdu.right()
        # dNdm(u, m; m_hat, v*)
        dNdm = dFdm.right()

        assert dNdu.derivatives == (1, 0)
        assert dNdm.derivatives == (0, 1)
        assert dNdu.arguments() == (vstar_N(0), u_hat(n_arg))
        assert dNdm.arguments() == (vstar_N(0), m_hat(n_arg))
        assert dNdu.argument_slots() == dNdu.arguments()
        assert dNdm.argument_slots() == dNdm.arguments()

        # Action
        action_dFdu = action(dFdu, w)
        action_dFdm = action(dFdm, p)

        assert action_dFdu.arguments() == v_F + ()
        assert action_dFdm.arguments() == v_F + ()

        # If we have 2 arguments
        if len(v_F):
            # Adjoint
            dFdu_adj = adjoint(dFdu)
            dFdm_adj = adjoint(dFdm)

            assert dFdu_adj.arguments() == (u_hat(n_arg),) + v_F
            assert dFdm_adj.arguments() == (m_hat(n_arg),) + v_F

            # Action of the adjoint
            q = Coefficient(v_F[0].ufl_function_space())
            action_dFdu_adj = action(dFdu_adj, q)
            action_dFdm_adj = action(dFdm_adj, q)

            assert action_dFdu_adj.arguments() == (u_hat(n_arg),)
            assert action_dFdm_adj.arguments() == (m_hat(n_arg),)


def test_multiple_external_operators(V1, V2):

    u = Coefficient(V1)
    m = Coefficient(V1)
    w = Coefficient(V2)

    v = TestFunction(V1)
    v_hat = TrialFunction(V1)
    w_hat = TrialFunction(V2)

    # N1(u, m; v*)
    N1 = ExternalOperator(u, m, function_space=V1)

    # N2(w; v*)
    N2 = ExternalOperator(w, function_space=V2)

    # N3(u; v*)
    N3 = ExternalOperator(u, function_space=V1)

    # N4(N1, u; v*)
    N4 = ExternalOperator(N1, u, function_space=V1)

    # N5(N4(N1, u); v*)
    N5 = ExternalOperator(N4, u, function_space=V1)

    # --- F = < N1(u, m; v*), v > + <N2(w; v*), v> + <N3(u; v*), v> --- #

    F = (inner(N1, v) + inner(N2, v) + inner(N3, v)) * dx

    # dFdu = Action(dFdN1, dN1du) + Action(dFdN3, dN3du)
    dFdu = expand_derivatives(derivative(F, u))
    dFdN1 = inner(v_hat, v) * dx
    dFdN2 = inner(w_hat, v) * dx
    dFdN3 = inner(v_hat, v) * dx
    dN1du = N1._ufl_expr_reconstruct_(u, m, derivatives=(1, 0), argument_slots=N1.arguments() + (v_hat,))
    dN3du = N3._ufl_expr_reconstruct_(u, derivatives=(1,), argument_slots=N3.arguments() + (v_hat,))

    assert dFdu == Action(dFdN1, dN1du) + Action(dFdN3, dN3du)

    # dFdm = Action(dFdN1, dN1dm)
    dFdm = expand_derivatives(derivative(F, m))
    dN1dm = N1._ufl_expr_reconstruct_(u, m, derivatives=(0, 1), argument_slots=N1.arguments() + (v_hat,))

    assert dFdm == Action(dFdN1, dN1dm)

    # dFdw = Action(dFdN2, dN2dw)
    dFdw = expand_derivatives(derivative(F, w))
    dN2dw = N2._ufl_expr_reconstruct_(w, derivatives=(1,), argument_slots=N2.arguments() + (w_hat,))

    assert dFdw == Action(dFdN2, dN2dw)

    # --- F = < N4(N1(u, m), u; v*), v > --- #

    F = inner(N4, v) * dx

    # dFdu = ∂F/∂u + Action(∂F/∂N1, dN1/du) + Action(∂F/∂N4, dN4/du)
    #      = Action(∂F/∂N4, dN4/du), since ∂F/∂u = 0 and ∂F/∂N1 = 0
    #
    # In addition, we have:
    # dN4/du = ∂N4/∂u + Action(∂N4/∂N1, dN1/du)
    #
    # Using the fact that Action is distributive, we have:
    #
    # dFdu = Action(∂F/∂N4, ∂N4/∂u) +
    #         Action(∂F/∂N4, Action(∂N4/∂N1, dN1/du))
    dFdu = expand_derivatives(derivative(F, u))
    dFdN4_partial = inner(v_hat, v) * dx
    dN4dN1_partial = N4._ufl_expr_reconstruct_(N1, u, derivatives=(1, 0), argument_slots=N4.arguments() + (v_hat,))
    dN4du_partial = N4._ufl_expr_reconstruct_(N1, u, derivatives=(0, 1), argument_slots=N4.arguments() + (v_hat,))

    assert dFdu == Action(dFdN4_partial, Action(dN4dN1_partial, dN1du)) + Action(dFdN4_partial, dN4du_partial)

    # dFdm = Action(∂F/∂N4, Action(∂N4/∂N1, dN1/dm))
    dFdm = expand_derivatives(derivative(F, m))

    assert dFdm == Action(dFdN4_partial, Action(dN4dN1_partial, dN1dm))

    # --- F = < N1(u, m; v*), v > + <N2(w; v*), v> + <N3(u; v*), v> + < N4(N1(u, m), u; v*), v > --- #

    F = (inner(N1, v) + inner(N2, v) + inner(N3, v) + inner(N4, v)) * dx

    dFdu = expand_derivatives(derivative(F, u))
    assert dFdu == Action(dFdN1, dN1du) + Action(dFdN3, dN3du) + Action(
        dFdN4_partial, Action(dN4dN1_partial, dN1du)) + Action(dFdN4_partial, dN4du_partial)

    dFdm = expand_derivatives(derivative(F, m))
    assert dFdm == Action(dFdN1, dN1dm) + Action(dFdN4_partial, Action(dN4dN1_partial, dN1dm))

    dFdw = expand_derivatives(derivative(F, w))
    assert dFdw == Action(dFdN2, dN2dw)

    # --- F = < N5(N4(N1(u, m), u), u; v*), v > + < N1(u, m; v*), v > + < u * N5(N4(N1(u, m), u), u; v*), v >--- #

    F = (inner(N5, v) + inner(N1, v) + inner(u * N5, v)) * dx

    # dFdu = ∂F/∂u + Action(∂F/∂N1, dN1/du) + Action(∂F/∂N4, dN4/du) + Action(∂F/∂N5, dN5/du)
    #
    # where:
    #  - ∂F/∂u = inner(w * N5, v) * dx
    #  - ∂F/∂N1 = inner(w, v) * dx
    #  - ∂F/∂N5 = inner(w, v) * dx + inner(u * w, v) * dx
    #  - ∂F/∂N4 = 0
    #  - dN5/du = ∂N5/∂u + Action(∂N5/∂N4, dN4/du)
    #           = ∂N5/∂u + Action(∂N5/∂N4, ∂N4/∂u) + Action(∂N5/∂N4, Action(∂N4/∂N1, dN1/du))
    # with w = TrialFunction(V1)
    w = TrialFunction(V1)
    dFdu_partial = inner(w * N5, v) * dx
    dFdN1_partial = inner(w, v) * dx
    dFdN5_partial = (inner(w, v) + inner(u * w, v)) * dx
    dN5dN4_partial = N5._ufl_expr_reconstruct_(N4, u, derivatives=(1, 0), argument_slots=N4.arguments() + (w,))
    dN5du_partial = N5._ufl_expr_reconstruct_(N4, u, derivatives=(0, 1), argument_slots=N4.arguments() + (w,))
    dN5du = Action(dN5dN4_partial, Action(dN4dN1_partial, dN1du)) + Action(
        dN5dN4_partial, dN4du_partial) + dN5du_partial

    dFdu = expand_derivatives(derivative(F, u))
    assert dFdu == dFdu_partial + Action(dFdN1_partial, dN1du) + Action(dFdN5_partial, dN5du)
