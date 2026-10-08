"""Test ExternalOperator object."""

__authors__ = "Nacime Bouziani"
__date__ = "2019-03-26"

import pytest
from utils import FiniteElement, LagrangeElement

from ufl import (
    Action,
    Argument,
    Coargument,
    Coefficient,
    Constant,
    Form,
    FunctionSpace,
    Matrix,
    Mesh,
    TestFunction,
    TrialFunction,
    action,
    adjoint,
    cos,
    derivative,
    ds,
    dx,
    inner,
    replace,
    sign,
    sin,
    triangle,
)
from ufl.algorithms import expand_derivatives
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.renumbering import renumber_indices
from ufl.coefficient import Cofunction
from ufl.constantvalue import Zero
from ufl.core.external_operator import ExternalOperator
from ufl.core.interpolate import Interpolate
from ufl.corealg.traversal import unique_pre_traversal
from ufl.differentiation import BaseFormDerivative, BaseFormOperatorDerivative
from ufl.form import BaseForm, ZeroBaseForm
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
    assert str(e) == "e(w_0; v_0)"

    e = ExternalOperator(u, function_space=V1, derivatives=(1,))
    assert str(e) == "∂e(w_0; v_0)/∂o1"

    e = ExternalOperator(
        u, r, 2 * s, t, function_space=V1, derivatives=(1, 0, 1, 2), argument_slots=(v0, v1)
    )
    assert str(e) == "∂e(w_0, w_1, 2 * w_2, w_3; v_1, v_0)/∂o1∂o3∂o4∂o4"


def test_form(V1, V2):
    u = Coefficient(V1)
    m = Coefficient(V1)
    u_hat = TrialFunction(V1)
    v = TestFunction(V1)

    # F = N * v * dx
    N = ExternalOperator(u, m, function_space=V2)
    F = N * v * dx
    actual = derivative(F, u, u_hat)

    (vstar,) = N.arguments()

    # dF/du[u_hat] = dN/du[u_hat] * v * dx
    dNdu = N._ufl_expr_reconstruct_(u, m, derivatives=(1, 0), argument_slots=(vstar, u_hat))
    assert apply_derivatives(actual) == dNdu * v * dx

    # F = N * u * v * dx
    N = ExternalOperator(u, m, function_space=V1)
    F = N * u * v * dx
    actual = derivative(F, u, u_hat)

    (vstar,) = N.arguments()

    # dF/du[u_hat] = (N * u_hat + dN/du[u_hat] * u) * v * dx
    dNdu = N._ufl_expr_reconstruct_(u, m, derivatives=(1, 0), argument_slots=(vstar, u_hat))
    assert apply_derivatives(actual) == (u_hat * N + u * dNdu) * v * dx


def test_form_dual_slot_argument_is_contracted(V1, V2):
    u = Coefficient(V1)
    v = TestFunction(V2)
    dual_slot = inner(u, v) * dx

    operator = ExternalOperator(u, function_space=V2, argument_slots=(dual_slot,))

    assert operator.arguments() == ()
    assert operator.ufl_element() == V2.ufl_element()


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
    (vstar_N1,) = N1.arguments()
    (vstar_N2,) = N2.arguments()
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
    (dN1du,) = Ja1.base_form_operators()
    dN1du_action = Action(dN1du, w)

    (dN2du,) = Ja2.base_form_operators()
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
    assert dN2du.argument_slots() == (vstar_dN2du, -sin(s) * s_hat)


def test_extractions(domain_2d, V1):
    from ufl.algorithms.analysis import (
        extract_arguments,
        extract_base_form_operators,
        extract_coefficients,
        extract_constants,
        extract_terminals_with_domain,
    )

    u = Coefficient(V1)
    c = Constant(domain_2d)

    e = ExternalOperator(u, c, function_space=V1)
    (vstar_e,) = e.arguments()

    assert extract_coefficients(e) == [u]
    assert extract_arguments(e) == [vstar_e]
    assert extract_terminals_with_domain(e) == ([vstar_e], [u], [])
    assert extract_constants(e) == [c]
    assert extract_base_form_operators(e) == [e]

    F = e * dx

    assert extract_coefficients(F) == [u]
    assert extract_arguments(e) == [vstar_e]
    assert extract_terminals_with_domain(e) == ([vstar_e], [u], [])
    assert extract_constants(F) == [c]
    assert F.base_form_operators() == (e,)

    u_hat = Argument(V1, 1)
    e = ExternalOperator(u, function_space=V1, derivatives=(1,), argument_slots=(vstar_e, u_hat))

    assert extract_coefficients(e) == [u]
    assert extract_arguments(e) == [vstar_e, u_hat]
    assert extract_terminals_with_domain(e) == ([vstar_e, u_hat], [u], [])
    assert extract_base_form_operators(e) == [e]

    F = e * dx

    assert extract_coefficients(F) == [u]
    assert extract_arguments(e) == [vstar_e, u_hat]
    assert extract_terminals_with_domain(e) == ([vstar_e, u_hat], [u], [])
    assert F.base_form_operators() == (e,)

    w = Coefficient(V1)
    e2 = ExternalOperator(w, e, function_space=V1)
    (vstar_e2,) = e2.arguments()

    assert extract_coefficients(e2) == [u, w]
    assert extract_arguments(e2) == [vstar_e2, u_hat]
    assert extract_terminals_with_domain(e2) == ([vstar_e2, u_hat], [u, w], [])
    assert extract_base_form_operators(e2) == [e, e2]

    F = e2 * dx

    assert extract_coefficients(e2) == [u, w]
    assert extract_arguments(e2) == [vstar_e2, u_hat]
    assert extract_terminals_with_domain(e2) == ([vstar_e2, u_hat], [u, w], [])
    assert F.base_form_operators() == (e, e2)


def get_external_operators(form_base):
    if isinstance(form_base, ExternalOperator):
        return (form_base,)
    elif isinstance(form_base, BaseForm):
        return form_base.base_form_operators()
    else:
        raise ValueError("Expecting BaseForm argument!")


def test_adjoint_action_jacobian(V1, V2, V3):
    u = Coefficient(V1)
    m = Coefficient(V2)

    # N(u, m; v*)
    N = ExternalOperator(u, m, function_space=V3)

    # Arguments for the Gateaux-derivative
    def u_hat(number):
        return Argument(V1, number)  # V1: degree 1 # dFdu.arguments()[-1]

    def m_hat(number):
        return Argument(V2, number)  # V2: degree 2 # dFdm.arguments()[-1]

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

        # dNdu(u, m; u_hat, v*)
        (dNdu,) = dFdu.base_form_operators()
        # dNdm(u, m; m_hat, v*)
        (dNdm,) = dFdm.base_form_operators()

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

            V = v_F[0].ufl_function_space()
            assert dFdu_adj.arguments() == (TestFunction(V1), TrialFunction(V))
            assert dFdm_adj.arguments() == (TestFunction(V2), TrialFunction(V))

            # Action of the adjoint
            q = Coefficient(V)
            action_dFdu_adj = action(dFdu_adj, q)
            action_dFdm_adj = action(dFdm_adj, q)

            assert action_dFdu_adj.arguments() == (TestFunction(V1),)
            assert action_dFdm_adj.arguments() == (TestFunction(V2),)


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

    # dFdu = < dN1/du, v > + < dN3/du, v >
    dFdu = expand_derivatives(derivative(F, u))
    dN1du = N1._ufl_expr_reconstruct_(
        u, m, derivatives=(1, 0), argument_slots=N1.arguments() + (v_hat,)
    )
    dN3du = N3._ufl_expr_reconstruct_(u, derivatives=(1,), argument_slots=N3.arguments() + (v_hat,))

    assert dFdu == apply_algebra_lowering((inner(dN1du, v) + inner(dN3du, v)) * dx)

    # dFdm = < dN1/dm, v >
    dFdm = expand_derivatives(derivative(F, m))
    dN1dm = N1._ufl_expr_reconstruct_(
        u, m, derivatives=(0, 1), argument_slots=N1.arguments() + (v_hat,)
    )

    assert dFdm == apply_algebra_lowering(inner(dN1dm, v) * dx)

    # dFdw = < dN2/dw, v >
    dFdw = expand_derivatives(derivative(F, w))
    dN2dw = N2._ufl_expr_reconstruct_(w, derivatives=(1,), argument_slots=N2.arguments() + (w_hat,))

    assert dFdw == apply_algebra_lowering(inner(dN2dw, v) * dx)

    # --- F = < N4(N1(u, m), u; v*), v > --- #

    F = inner(N4, v) * dx

    # dFdu = < dN4/du, v >, where the chain rule gives
    # dN4/du = ∂N4/∂N1[dN1/du] + ∂N4/∂u
    def dN4dN1(dN1):
        return N4._ufl_expr_reconstruct_(
            N1, u, derivatives=(1, 0), argument_slots=N4.arguments() + (dN1,)
        )

    dN4du_partial = N4._ufl_expr_reconstruct_(
        N1, u, derivatives=(0, 1), argument_slots=N4.arguments() + (v_hat,)
    )
    dN4du = dN4dN1(dN1du) + dN4du_partial

    dFdu = expand_derivatives(derivative(F, u))
    assert dFdu == apply_algebra_lowering(inner(dN4du, v) * dx)

    # dFdm = < ∂N4/∂N1[dN1/dm], v >
    dFdm = expand_derivatives(derivative(F, m))
    assert dFdm == apply_algebra_lowering(inner(dN4dN1(dN1dm), v) * dx)

    # --- F = < N1(u, m; v*), v > + <N2(w; v*), v> + <N3(u; v*), v> + <
    # N4(N1(u, m), u; v*), v > --- #

    F = (inner(N1, v) + inner(N2, v) + inner(N3, v) + inner(N4, v)) * dx

    dFdu = expand_derivatives(derivative(F, u))
    assert dFdu == apply_algebra_lowering(
        (inner(dN1du, v) + inner(dN3du, v) + inner(dN4du, v)) * dx
    )

    dFdm = expand_derivatives(derivative(F, m))
    assert dFdm == apply_algebra_lowering((inner(dN1dm, v) + inner(dN4dN1(dN1dm), v)) * dx)

    dFdw = expand_derivatives(derivative(F, w))
    assert dFdw == apply_algebra_lowering(inner(dN2dw, v) * dx)

    # --- F = < N5(N4(N1(u, m), u), u; v*), v > + < N1(u, m; v*), v > +
    # < u * N5(N4(N1(u, m), u), u; v*), v >--- #

    F = (inner(N5, v) + inner(N1, v) + inner(u * N5, v)) * dx

    # dFdu = < dN5/du, v > + < dN1/du, v > + < v_hat * N5 + u * dN5/du, v >,
    # where the chain rule gives dN5/du = ∂N5/∂N4[dN4/du] + ∂N5/∂u
    dN5dN4 = N5._ufl_expr_reconstruct_(
        N4, u, derivatives=(1, 0), argument_slots=N5.arguments() + (dN4du,)
    )
    dN5du_partial = N5._ufl_expr_reconstruct_(
        N4, u, derivatives=(0, 1), argument_slots=N5.arguments() + (v_hat,)
    )
    dN5du = dN5dN4 + dN5du_partial

    dFdu = expand_derivatives(derivative(F, u))
    assert dFdu == apply_algebra_lowering(
        (inner(dN5du, v) + inner(dN1du, v) + inner(v_hat * N5 + u * dN5du, v)) * dx
    )


def test_replace(V1):
    u = Coefficient(V1, count=0)
    N = ExternalOperator(u, function_space=V1)

    # dN(u; uhat, v*)
    dN = expand_derivatives(derivative(N, u))
    vstar, uhat = dN.arguments()
    assert isinstance(vstar, Coargument)

    # Replace v* by a Form
    v = TestFunction(V1)
    F = inner(u, v) * dx
    G = replace(dN, {vstar: F})

    dN_replaced = dN._ufl_expr_reconstruct_(u, argument_slots=(F, uhat))
    assert G == dN_replaced

    # Replace v* by an Action
    M = Matrix(V1, V1)
    A = Action(M, u)
    G = replace(dN, {vstar: A})

    dN_replaced = dN._ufl_expr_reconstruct_(u, argument_slots=(A, uhat))
    assert G == dN_replaced


def test_replace_base_form_operator(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    M = ExternalOperator(w, function_space=V1)
    Iw = Interpolate(w, V1)
    e = N * v + M * v + Iw * v

    # Replacing N by zero drops its term and keeps the operators that don't depend on N.
    # Check them before comparing r, since == shares the operands of equal expressions.
    r = replace(e, {N: Zero()})
    operators = [
        o for o in unique_pre_traversal(r) if isinstance(o, ExternalOperator | Interpolate)
    ]
    assert len(operators) == 2
    assert all(o is M or o is Iw for o in operators)
    assert r == M * v + Iw * v


def test_replace_base_form_derivative(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(Action(N * u * v * dx, u), u)
    assert isinstance(dJ, BaseFormDerivative)

    with pytest.raises(ValueError, match="Derivatives should be applied"):
        replace(dJ, {u: w})


def test_base_form_derivative_lowers_compound_algebra(domain_2d):
    V = FunctionSpace(domain_2d, FiniteElement("CG", triangle, 1, (2,), identity_pullback, H1))
    u = Coefficient(V)
    v = TestFunction(V)
    N = ExternalOperator(u, function_space=V)
    F = inner(N, v) * dx

    def dJ(F):
        return renumber_indices(expand_derivatives(derivative(Action(F, u), u)))

    assert dJ(F) == dJ(apply_algebra_lowering(F))


def test_ZeroDerivative(V1):
    u = Coefficient(V1, count=1)
    N = ExternalOperator(Coefficient(V1, count=0), function_space=V1)
    dN1 = expand_derivatives(derivative(N, u))
    assert isinstance(dN1, ZeroBaseForm)


def test_dual_slot_derivative(V1, V2):
    u = Coefficient(V1)
    u_hat = Argument(V1, 1)
    v = TestFunction(V2)
    vstar = inner(u, v) * dx
    N = ExternalOperator(u, function_space=V2, argument_slots=(vstar,))

    dNdu = ExternalOperator(
        u,
        function_space=V2,
        derivatives=(1,),
        argument_slots=(vstar, u_hat),
    )
    # N is linear in its dual slot, so the product rule acts with N(u; vhat) on dv*.
    vhat = Coargument(V2.dual(), 0)
    N_vhat = ExternalOperator(u, function_space=V2, argument_slots=(vhat,))
    expected = dNdu + Action(N_vhat, inner(u_hat, v) * dx)

    dN = derivative(N, u, u_hat)
    assert isinstance(dN, BaseFormOperatorDerivative)
    assert expand_derivatives(dN) == expected


def test_dual_slot_second_derivative(V1, V2):
    u = Coefficient(V1)
    u1, u2 = Argument(V1, 1), Argument(V1, 2)
    v = TestFunction(V2)
    vstar = inner(u, v) * dx
    N = ExternalOperator(u, function_space=V2, argument_slots=(vstar,))

    def dN(slots, n):
        return ExternalOperator(u, function_space=V2, derivatives=(n,), argument_slots=slots)

    # D^2 N(u; v*(u))[u1, u2] = d2N(u; v*)[u1, u2] + dN(u; Dv*[u2])[u1] + dN(u; Dv*[u1])[u2]
    vhat2, vhat3 = Coargument(V2.dual(), 2), Coargument(V2.dual(), 3)
    expected = {
        dN((vstar, u1, u2), 2),
        Action(dN((vhat2, u1), 1), inner(u2, v) * dx),
        Action(dN((vhat3, u2), 1), inner(u1, v) * dx),
    }
    d2N = expand_derivatives(derivative(derivative(N, u, u1), u, u2))
    assert set(d2N.components()) == expected
    assert all(c.arguments() == (u1, u2) for c in d2N.components())


def test_chain_rule_skips_underived_integrals(V1):
    u = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(N**2 * dx, u)
    F = u * v * dx

    # F is not differentiated, so it does not contribute to dJ/dN.
    assert expand_derivatives(dJ + F) == expand_derivatives(dJ) + F


def test_action_derivative_wrt_base_form_operator(V1):
    u = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    A = Action(inner(u, v) * dx, N)

    # N is not a coefficient of A, but A depends on N through the right slot.
    dA = derivative(A, N, v)
    # The Leibniz rule is applied when the derivative is expanded.
    assert isinstance(dA, BaseFormDerivative)
    assert expand_derivatives(dA) == inner(u, v) * dx


def test_chain_rule_for_each_derivative(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    uhat = TrialFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    M = ExternalOperator(w, function_space=V1)
    dF = derivative(N * v * dx, u, uhat)
    dG = derivative(M * v * ds, w, uhat)

    # Each derivative applies the chain rule through its own base form operators.
    assert expand_derivatives(dF + dG) == expand_derivatives(dF) + expand_derivatives(dG)


def test_operator_nested_in_slots_of_slots(V1):
    u = Coefficient(V1)
    v = TestFunction(V1)
    N1 = ExternalOperator(u, function_space=V1)
    N3 = ExternalOperator(ExternalOperator(N1, function_space=V1), function_space=V1)

    # dN1/du is in the integrand and in the slot of dN2/dN1 in the slot of dN3/dN2.
    dF = expand_derivatives(derivative(N1 * v * dx + N3 * v * dx, u))
    dF1 = expand_derivatives(derivative(N1 * v * dx, u))
    dF3 = expand_derivatives(derivative(N3 * v * dx, u))
    assert dF == dF1 + dF3


def test_chain_rule_only_differentiates_its_derivative(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    uhat = TrialFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dF = derivative(N * w * v * dx, u, uhat)
    dG = derivative(w**2 * N * v * ds, w, uhat)

    # dG depends on N(u), but it is not differentiated with respect to u.
    assert expand_derivatives(dF + dG) == expand_derivatives(dF) + expand_derivatives(dG)


def test_action_derivative_through_composition(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    uhat = TrialFunction(V1)
    M = ExternalOperator(u, function_space=V1)
    N = ExternalOperator(M, function_space=V1)
    F = w * v * dx

    # D[Action(F, N(M(u)))] = Action(F, dN/dM[dM/du])
    dMdu = M._ufl_expr_reconstruct_(u, derivatives=(1,), argument_slots=M.arguments() + (uhat,))
    dNdM = N._ufl_expr_reconstruct_(M, derivatives=(1,), argument_slots=N.arguments() + (dMdu,))
    dA = expand_derivatives(derivative(Action(F, N), u, uhat))
    assert dA == Action(F, dNdM)


def test_vanishing_derivative(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    uhat = TrialFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    M = ExternalOperator(w, function_space=V1)

    # The derivative keeps its arguments when it vanishes.
    assert expand_derivatives(derivative(sign(N) * v * dx, u, uhat)) == ZeroBaseForm((v, uhat))
    assert expand_derivatives(derivative(M * v * dx, u, uhat)) == ZeroBaseForm((v, uhat))

    # An empty Form has no derivative to vanish.
    assert type(apply_derivatives(Form([]))) is Form


def test_coefficient_derivatives(V1):
    u = Coefficient(V1)
    g = Coefficient(V1)
    v = TestFunction(V1)
    uhat = TrialFunction(V1)
    N = ExternalOperator(u, function_space=V1)

    # The given derivative of N replaces the derivative of its operator.
    dF = derivative(N * v * dx, u, uhat, coefficient_derivatives={N: g})
    assert expand_derivatives(dF) == g * uhat * v * dx


def test_functional_derivative(V1):
    u = Coefficient(V1)
    v0 = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    J = N**2 * dx

    # dJ/du[v0] = 2 * N * dN/du[v0] * dx
    dNdu = N._ufl_expr_reconstruct_(u, derivatives=(1,), argument_slots=N.arguments() + (v0,))
    dJdu = expand_derivatives(derivative(J, u))
    assert dJdu == 2 * dNdu * N * dx
    assert dJdu.arguments() == (v0,)


def test_functional_hessian_through_composition(V1):
    u = Coefficient(V1)
    M = ExternalOperator(u, function_space=V1)
    N = ExternalOperator(M, function_space=V1)
    H = derivative(derivative(N * dx, u), u)

    def a(number):
        return Argument(V1, number)

    def c(number):
        return Coargument(V1.dual(), number)

    def dM(n, *slots):
        return M._ufl_expr_reconstruct_(u, derivatives=(n,), argument_slots=slots)

    def dN(n, *slots):
        return N._ufl_expr_reconstruct_(M, derivatives=(n,), argument_slots=slots)

    # H[v0, v1] = d2N/dM2[dM/du[v0], dM/du[v1]] + dN/dM[d2M/du2[v0, v1]]
    d2N = dN(2, c(0), dM(1, c(0), a(0)), dM(1, c(0), a(1)))
    d2M = dN(1, c(0), dM(2, c(0), a(0), a(1)))
    assert expand_derivatives(H) == (d2N + d2M) * dx


def test_extraction_external_operator_composition(V1, V2, V3, V4, V5):
    from ufl.algorithms.analysis import extract_arguments

    u5 = Coefficient(V5)
    u4 = ExternalOperator(u5, function_space=V4)
    u3 = ExternalOperator(u4, function_space=V3)
    u2 = ExternalOperator(u3, function_space=V2)
    u1 = ExternalOperator(u2, function_space=V1)

    assert u4.ufl_function_space() == V4
    assert u3.ufl_function_space() == V3
    assert u2.ufl_function_space() == V2
    assert u1.ufl_function_space() == V1

    u1 = Cofunction(V1.dual())
    arg2 = Argument(V2, 0)
    e1 = ExternalOperator(arg2, function_space=V1, argument_slots=(u1, arg2))

    arg3 = Argument(V3, 0)
    e2 = ExternalOperator(arg3, function_space=V2, argument_slots=(e1, arg3))

    arg4 = Argument(V4, 0)
    e3 = ExternalOperator(arg4, function_space=V3, argument_slots=(e2, arg4))

    arg5 = Argument(V5, 0)
    e4 = ExternalOperator(arg5, function_space=V4, argument_slots=(e3, arg5))

    args = extract_arguments(e4)

    assert e1.ufl_function_space() == V2.dual()
    assert e2.ufl_function_space() == V3.dual()
    assert e3.ufl_function_space() == V4.dual()
    assert e4.ufl_function_space() == V5.dual()

    assert set(args) == {arg2, arg3, arg4, arg5, Argument(V1, 0)}
