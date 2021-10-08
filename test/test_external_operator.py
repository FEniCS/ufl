#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

__authors__ = "Nacime Bouziani"
__date__ = "2019-03-26"


"""
Test ExternalOperator object
"""

import pytest

# This imports everything external code will see from ufl
from ufl import *
from ufl.core.external_operator import ExternalOperator
from ufl.form import BaseForm
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms import expand_derivatives
from ufl.constantvalue import as_ufl
from ufl.domain import default_domain

from functools import partial


@pytest.fixture(scope='module')
def V1():
    return FiniteElement("CG", triangle, 1)


@pytest.fixture(scope='module')
def V2():
    return FiniteElement("CG", triangle, 2)


@pytest.fixture(scope='module')
def V3():
    return FiniteElement("CG", triangle, 3)


def test_properties(self, cell):
    S = FiniteElement("CG", cell, 1)
    u = Coefficient(S, count=0)
    r = Coefficient(S, count=1)

    e = ExternalOperator(u, r, function_space=S)

    domain = default_domain(cell)
    space = FunctionSpace(domain, S)

    assert e.ufl_function_space() == space
    assert e.ufl_operands[0] == u
    assert e.ufl_operands[1] == r
    assert e.derivatives == (0, 0)
    assert e.ufl_shape == ()

    e2 = ExternalOperator(u, r, function_space=S, derivatives=(3, 4))
    assert e2.derivatives == (3, 4)
    assert e2.ufl_shape == ()

    # Test __str__
    s = Coefficient(S, count=2)
    t = Coefficient(S, count=3)
    v0 = Argument(S, 0)
    v1 = Argument(S, 1)

    e = ExternalOperator(u, function_space=S)
    assert str(e) == 'e(w_0; v_0)'

    e = ExternalOperator(u, function_space=S, derivatives=(1,))
    assert str(e) == '∂e(w_0; v_0)/∂o1'

    e = ExternalOperator(u, r, 2 * s, t, function_space=S, derivatives=(1, 0, 1, 2), argument_slots=(v0, v1))
    assert str(e) == '∂e(w_0, w_1, 2 * w_2, w_3; v_1, v_0)/∂o1∂o3∂o4∂o4'


def _make_external_operator(V=None, nops=1):
    space = V or FiniteElement("Quadrature", triangle, 1)
    return ExternalOperator(*[variable(0.) for _ in range(nops)], function_space=space)


def _test(f, df, V):
    v = Coefficient(V)
    fexpr = f(v)

    dfv1 = diff(fexpr, v)
    dfv2 = df(v)
    assert apply_derivatives(dfv1) == dfv2


def _test_multivariable(f, df1, df2, df3, V):
    v1 = Coefficient(V)
    v2 = Coefficient(V)
    v3 = Coefficient(V)
    fexpr = f(v1, v2, v3)

    dfv1 = diff(fexpr, v1)
    dfv2 = df1(v1, v2, v3)
    assert apply_derivatives(dfv1) == dfv2

    dfv1 = diff(fexpr, v2)
    dfv2 = df2(v1, v2, v3)
    assert apply_derivatives(dfv1) == dfv2

    dfv1 = diff(fexpr, v3)
    dfv2 = df3(v1, v2, v3)
    assert apply_derivatives(dfv1) == dfv2


def testVariable(V1):
    e = _make_external_operator(V1)

    def f(v, e):
        nl = e._ufl_expr_reconstruct_(v, derivatives=(0,))
        return as_ufl(nl)

    def df(v, e):
        nl = e._ufl_expr_reconstruct_(v, derivatives=(1,))
        return as_ufl(nl)

    def df2(v, e):
        nl = e._ufl_expr_reconstruct_(v, derivatives=(2,))
        return as_ufl(nl)

    fe = partial(f, e=e)
    dfe = partial(df, e=e)
    df2e = partial(df2, e=e)
    _test(fe, dfe, V1)
    _test(dfe, df2e, V1)


def testProduct(V1):
    e = _make_external_operator(V1)

    def g(v, e):
        nl = e._ufl_expr_reconstruct_(v, derivatives=(0,))
        return as_ufl(nl)

    def f(v, e):
        return 3 * g(v, e)

    def df(v, e):
        e = g(v, e)
        nl = e._ufl_expr_reconstruct_(v, derivatives=(1,))
        return as_ufl(3 * nl)

    fe = partial(f, e=e)
    dfe = partial(df, e=e)
    _test(fe, dfe, V1)


def testProductExternalOperator(V1):
    e1 = _make_external_operator(V1)
    e2 = _make_external_operator(V1)

    cst = 2.0

    def g(v, e1, e2):
        nl = e1._ufl_expr_reconstruct_(cst * v)
        nl2 = e2._ufl_expr_reconstruct_(v)
        return nl, nl2

    def f(v, e1, e2):
        eo1, eo2 = g(v, e1, e2)
        return eo1 * eo2

    def df(v, e1, e2):
        nl = e1._ufl_expr_reconstruct_(cst * v)
        nl2 = e2._ufl_expr_reconstruct_(v)
        dnl = cst * e1._ufl_expr_reconstruct_(cst * v, derivatives=(1,))
        dnl2 = e2._ufl_expr_reconstruct_(v, derivatives=(1,))

        return as_ufl(dnl * nl2 + dnl2 * nl)

    fe = partial(f, e1=e1, e2=e2)
    dfe = partial(df, e1=e1, e2=e2)
    _test(fe, dfe, V1)


def testmultiVariable(V1):
    e = _make_external_operator(V1, 3)

    def g(v1, v2, v3, e):
        return e._ufl_expr_reconstruct_(v1, v2, v3)

    def f(v1, v2, v3, e):
        return cos(v1) * sin(v2) * g(v1, v2, v3, e)

    def df1(v1, v2, v3, e):
        r = g(v1, v2, v3, e)
        g1 = r._ufl_expr_reconstruct_(v1, v2, v3, derivatives=(0, 0, 0))
        g2 = r._ufl_expr_reconstruct_(v1, v2, v3, derivatives=(1, 0, 0))
        nl = - sin(v1) * sin(v2) * g1 + cos(v1) * sin(v2) * g2
        return as_ufl(nl)

    def df2(v1, v2, v3, e):
        r = g(v1, v2, v3, e)
        g1 = r._ufl_expr_reconstruct_(v1, v2, v3, derivatives=(0, 0, 0))
        g2 = r._ufl_expr_reconstruct_(v1, v2, v3, derivatives=(0, 1, 0))
        nl = cos(v2) * cos(v1) * g1 + cos(v1) * sin(v2) * g2
        return as_ufl(nl)

    def df3(v1, v2, v3, e):
        r = g(v1, v2, v3, e)
        g1 = r._ufl_expr_reconstruct_(v1, v2, v3, derivatives=(0, 0, 1))
        nl = cos(v1) * sin(v2) * g1
        return as_ufl(nl)

    fe = partial(f, e=e)
    df1e = partial(df1, e=e)
    df2e = partial(df2, e=e)
    df3e = partial(df3, e=e)
    _test_multivariable(fe, df1e, df2e, df3e, V1)


def test_form():
    cell = triangle
    V = FiniteElement("CG", cell, 1)
    P = FiniteElement("Quadrature", cell, 2)
    u = Coefficient(V)
    m = Coefficient(V)
    u_hat = TrialFunction(V)
    v = TestFunction(V)

    # F = N * v * dx
    N = ExternalOperator(u, m, function_space=P)
    F = N * v * dx
    actual = derivative(F, u, u_hat)

    vstar, = N.arguments()
    Nhat = TrialFunction(N.ufl_function_space())

    dNdu = N._ufl_expr_reconstruct_(u, m, derivatives=(1, 0), argument_slots=(vstar, u_hat))
    dFdN = Nhat * v * dx
    expected = Action(dFdN, dNdu)

    assert apply_derivatives(actual) == expected

    # F = N * u * v * dx
    N = ExternalOperator(u, m, function_space=V)
    F = N * u * v * dx
    actual = derivative(F, u, u_hat)

    vstar, = N.arguments()
    Nhat = TrialFunction(N.ufl_function_space())

    dNdu = N._ufl_expr_reconstruct_(u, m, derivatives=(1, 0), argument_slots=(vstar, u_hat))
    dFdu_partial = N * u_hat * v * dx
    dFdN = Nhat * u * v * dx
    expected = dFdu_partial + Action(dFdN, dNdu)
    assert apply_derivatives(actual) == expected


def test_function_spaces_derivatives():
    V = FiniteElement("CG", triangle, 1)
    Vv = VectorElement("CG", triangle, 1)
    Vt = TensorElement("CG", triangle, 1)
    Vt2 = TensorElement(V, shape=(2, 2, 2))
    Vt3 = TensorElement(V, shape=(2, 2, 2, 2))
    Vt4 = TensorElement(V, shape=(2, 2, 2, 2, 2))
    Vt5 = TensorElement(V, shape=(2, 2, 2, 2, 2, 2))

    def _check_space_shape_fct_space(x, der, shape, space):
        assert x.derivatives == der
        assert x.ufl_shape == shape
        assert x.ufl_function_space().ufl_element() == space

    u = Coefficient(V)
    w = Coefficient(V)

    uv = Coefficient(Vv)
    ut = Coefficient(Vt)

    # Scalar case

    e = ExternalOperator(u, w, function_space=V)
    dedu = e._ufl_expr_reconstruct_(u, w, derivatives=(1, 0))
    dedw = e._ufl_expr_reconstruct_(u, w, derivatives=(0, 1))
    d2edu = e._ufl_expr_reconstruct_(u, w, derivatives=(2, 0))
    dedwdu = e._ufl_expr_reconstruct_(u, w, derivatives=(1, 1))
    d2edw = e._ufl_expr_reconstruct_(u, w, derivatives=(0, 2))

    _check_space_shape_fct_space(dedu, (1, 0), (), V)
    _check_space_shape_fct_space(dedw, (0, 1), (), V)

    _check_space_shape_fct_space(d2edu, (2, 0), (), V)
    _check_space_shape_fct_space(dedwdu, (1, 1), (), V)
    _check_space_shape_fct_space(d2edw, (0, 2), (), V)

    # Vector case
    ev = ExternalOperator(uv, w, function_space=Vv)
    deduv = ev._ufl_expr_reconstruct_(uv, w, derivatives=(1, 0))
    dedw = ev._ufl_expr_reconstruct_(uv, w, derivatives=(0, 1))
    d2eduv = ev._ufl_expr_reconstruct_(uv, w, derivatives=(2, 0))
    dedwduv = ev._ufl_expr_reconstruct_(uv, w, derivatives=(1, 1))
    d2edw = ev._ufl_expr_reconstruct_(uv, w, derivatives=(0, 2))

    _check_space_shape_fct_space(deduv, (1, 0), (2, 2), Vt)
    _check_space_shape_fct_space(dedw, (0, 1), (2,), Vv)

    _check_space_shape_fct_space(d2eduv, (2, 0), (2, 2, 2), Vt2)
    _check_space_shape_fct_space(dedwduv, (1, 1), (2, 2), Vt)
    _check_space_shape_fct_space(d2edw, (0, 2), (2,), Vv)

    # Tensor case
    et = ExternalOperator(ut, uv, w, function_space=Vt)
    dedut = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(1, 0, 0))
    deduv = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 1, 0))
    dedw = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 0, 1))

    _check_space_shape_fct_space(dedut, (1, 0, 0), (2, 2, 2, 2), Vt3)
    _check_space_shape_fct_space(deduv, (0, 1, 0), (2, 2, 2), Vt2)
    _check_space_shape_fct_space(dedw, (0, 0, 1), (2, 2), Vt)

    d2edut = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(2, 0, 0))
    d2eduv = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 2, 0))
    d2edw = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 0, 2))

    _check_space_shape_fct_space(d2edut, (2, 0, 0), (2, 2, 2, 2, 2, 2), Vt5)
    _check_space_shape_fct_space(d2eduv, (0, 2, 0), (2, 2, 2, 2), Vt3)
    _check_space_shape_fct_space(d2edw, (0, 0, 2), (2, 2), Vt)

    dedwduv = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 1, 1))
    dedwdut = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(1, 0, 1))
    dedutduv = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(1, 1, 0))

    _check_space_shape_fct_space(dedwduv, (0, 1, 1), (2, 2, 2), Vt2)
    _check_space_shape_fct_space(dedwdut, (1, 0, 1), (2, 2, 2, 2), Vt3)
    _check_space_shape_fct_space(dedutduv, (1, 1, 0), (2, 2, 2, 2, 2), Vt4)

    # TODO: MIXED ELEMENT


def test_differentiation_procedure_action():
    S = FiniteElement("CG", triangle, 1)
    V = VectorElement("CG", triangle, 1)
    s = Coefficient(S)
    u = Coefficient(V)
    m = Coefficient(V)

    # External operators
    N1 = ExternalOperator(u, m, function_space=V)
    N2 = ExternalOperator(cos(s), function_space=V)

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
    assert vstar_N1.ufl_function_space().ufl_element() == V
    assert vstar_N2.ufl_function_space().ufl_element() == V

    # The external operators have an argument vstar with number 1
    u_hat = Argument(V, 2)
    s_hat = Argument(S, 2)
    w = Coefficient(V)
    r = Coefficient(S)

    # Bilinear forms
    a1 = inner(N1, m) * dx
    Ja1 = derivative(a1, u, u_hat)
    Ja1 = expand_derivatives(Ja1)

    a2 = inner(N2, m) * dx
    Ja2 = derivative(a2, s, s_hat)
    Ja2 = expand_derivatives(Ja2)

    assert len(Ja1.external_operators()) == 1
    assert len(Ja2.external_operators()) == 1

    # Get external operators
    dN1du, = Ja1.external_operators()
    dN1du_action = Action(dN1du, w)

    dN2du, = Ja2.external_operators()
    dN2du_action = Action(dN2du, r)

    # Check shape
    assert dN1du.ufl_shape == (2,)
    assert dN2du.ufl_shape == (2,)

    # Get v*s
    vstar_dN1du, _ = dN1du.arguments()
    vstar_dN2du, _ = dN2du.arguments()
    assert vstar_dN1du.ufl_function_space().ufl_element() == V  # shape: (2,)
    assert vstar_dN2du.ufl_function_space().ufl_element() == V  # shape: (2,)

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


def test_extractions():
    from ufl.algorithms.analysis import (extract_coefficients, extract_arguments,
                                         extract_arguments_and_coefficients,
                                         extract_external_operators, extract_constants)

    V = FiniteElement("CG", triangle, 1)
    u = Coefficient(V)
    c = Constant(triangle)

    e = ExternalOperator(u, c, function_space=V)
    vstar_e, = e.arguments()

    assert extract_coefficients(e) == [u, e.result_coefficient()]
    assert extract_arguments(e) == [vstar_e]
    assert extract_arguments_and_coefficients(e) == ([vstar_e], [u, e.result_coefficient()])
    assert extract_constants(e) == [c]
    assert extract_external_operators(e) == [e]

    F = e * dx

    assert extract_coefficients(F) == [u, e.result_coefficient()]
    assert extract_arguments(e) == [vstar_e]
    assert extract_arguments_and_coefficients(e) == ([vstar_e], [u, e.result_coefficient()])
    assert extract_constants(F) == [c]
    assert F.external_operators() == (e,)

    # The external operators have an argument vstar with number 1
    u_hat = Argument(V, 2)
    e = ExternalOperator(u, function_space=V, derivatives=(1,), argument_slots=(vstar_e, u_hat))

    assert extract_coefficients(e) == [u, e.result_coefficient()]
    assert extract_arguments(e) == [vstar_e, u_hat]
    assert extract_arguments_and_coefficients(e) == ([vstar_e, u_hat], [u, e.result_coefficient()])
    assert extract_external_operators(e) == [e]

    F = e * dx

    assert extract_coefficients(F) == [u, e.result_coefficient()]
    assert extract_arguments(e) == [vstar_e, u_hat]
    assert extract_arguments_and_coefficients(e) == ([vstar_e, u_hat], [u, e.result_coefficient()])
    assert F.external_operators() == (e,)

    w = Coefficient(V)
    e2 = ExternalOperator(w, e, function_space=V)
    vstar_e2, = e2.arguments()

    assert extract_coefficients(e2) == [u, e.result_coefficient(), w, e2.result_coefficient()]
    assert extract_arguments(e2) == [vstar_e2, u_hat]
    assert extract_arguments_and_coefficients(e2) == ([vstar_e2, u_hat], [u, e.result_coefficient(), w, e2.result_coefficient()])
    assert extract_external_operators(e2) == [e, e2]

    F = e2 * dx

    assert extract_coefficients(e2) == [u, e.result_coefficient(), w, e2.result_coefficient()]
    assert extract_arguments(e2) == [vstar_e2, u_hat]
    assert extract_arguments_and_coefficients(e2) == ([vstar_e2, u_hat], [u, e.result_coefficient(), w, e2.result_coefficient()])
    assert F.external_operators() == (e, e2)


def get_external_operators(form_base):
    if isinstance(form_base, ExternalOperator):
        return (form_base,)
    elif isinstance(form_base, BaseForm):
        return form_base.external_operators()
    else:
        raise ValueError('Expecting FormBase argument!')


def test_adjoint_action_jacobian(V1, V2, V3):

    u = Coefficient(V1)
    m = Coefficient(V2)

    # N(u, m; v*)
    N = ExternalOperator(u, m, function_space=V3)
    vstar_N, = N.arguments()

    # Arguments for the Gateaux-derivative
    u_hat = lambda number: Argument(V1, number)   # V1: degree 1 # dFdu.arguments()[-1]
    m_hat = lambda number: Argument(V2, number)   # V2: degree 2 # dFdm.arguments()[-1]
    vstar_N = lambda number: Coargument(V3, number)  # V3: degree 3

    # Coefficients for the action
    w = Coefficient(V1)  # for u
    p = Coefficient(V2)  # for m

    v2 = TestFunction(V2)
    v3 = TestFunction(V3)
    form_base_expressions = (N * dx, N * v2 * dx, N * v3 * dx)  # , N)

    for F in form_base_expressions:

        # Get test function
        v_F = F.arguments() if isinstance(F, Form) else ()
        n_arg = len(v_F)
        assert n_arg < 2

        # Differentiate
        dFdu = expand_derivatives(derivative(F, u, u_hat(n_arg + 1)))
        dFdm = expand_derivatives(derivative(F, m, m_hat(n_arg + 1)))

        assert dFdu.arguments() == v_F + (u_hat(n_arg + 1),)
        assert dFdm.arguments() == v_F + (m_hat(n_arg + 1),)

        # dNdu(u, m; u_hat, v*)
        dNdu, = dFdu.external_operators()
        # dNdm(u, m; m_hat, v*)
        dNdm, = dFdm.external_operators()

        assert dNdu.derivatives == (1, 0)
        assert dNdm.derivatives == (0, 1)
        assert dNdu.arguments() == (vstar_N(0), u_hat(n_arg + 1))
        assert dNdm.arguments() == (vstar_N(0), m_hat(n_arg + 1))
        assert dNdu.argument_slots() == dNdu.arguments()
        assert dNdm.argument_slots() == dNdm.arguments()

        # Action
        action_dFdu = action(dFdu, w)
        action_dFdm = action(dFdm, p)

        assert action_dFdu.arguments() == v_F + ()
        assert action_dFdm.arguments() == v_F + ()

        # If we have 2 arguments
        if n_arg > 0:
            # Adjoint
            dFdu_adj = adjoint(dFdu)
            dFdm_adj = adjoint(dFdm)

            assert dFdu_adj.arguments() == (u_hat(n_arg + 1),) + v_F
            assert dFdm_adj.arguments() == (m_hat(n_arg + 1),) + v_F

            # Action of the adjoint
            q = Coefficient(v_F[0].ufl_function_space())
            action_dFdu_adj = action(dFdu_adj, q)
            action_dFdm_adj = action(dFdm_adj, q)

            assert action_dFdu_adj.arguments() == (u_hat(n_arg + 1),)
            assert action_dFdm_adj.arguments() == (m_hat(n_arg + 1),)


"""
def test_adjoint_action_hessian():

    V = FiniteElement("CG", triangle, 1)
    u = Coefficient(V)
    m = Coefficient(V)

    # N(u, m; v*)
    N = ExternalOperator(u, m, function_space=V)
    vstar_N, = N.arguments()

    # Arguments for the Gateaux-derivatives (same function space for sake of simplicity)
    v1 = Argument(V, 1)  # for the first derivative
    v2 = Argument(V, 2)  # for the second derivative

    # Coefficients for the action
    q = Coefficient(V)  # for N
    w = Coefficient(V)  # for u
    p = Coefficient(V)  # for m

    # v2 = TestFunction(V2)
    # v3 = TestFunction(V3)
    form_base_expressions = (N * dx,)  # , N*v2*dx, N*v3*dx, N)
    for F in form_base_expressions:

        dFdu = derivative(F, u)
        dFdm = derivative(F, m)

        # Second derivative
        d2Fdu = expand_derivatives(derivative(dFdu, u))
        d2Fdm = expand_derivatives(derivative(dFdm, m))
        d2Fdmdu = expand_derivatives(derivative(dFdm, u))
        d2Fdudm = expand_derivatives(derivative(dFdu, m))

        def _check_second_derivative(d2F, derivatives, arguments, argument_slots=None):
            # Get the external operator
            d2N, = get_external_operators(d2F)
            assert d2N.derivatives == derivatives
            assert d2N.arguments() == arguments
            assert d2N.argument_slots() == argument_slots or arguments

        # d2Ndu(u, m; v2, v1, v*)
        _check_second_derivative(d2Fdu, (2, 0), (vstar_N, v1, v2))
        # d2Ndm(u, m; v2, v1, v*)
        _check_second_derivative(d2Fdm, (0, 2), (vstar_N, v1, v2))
        # d2Ndmdu(u, m; v2, v1, v*)
        _check_second_derivative(d2Fdmdu, (1, 1), (vstar_N, v1, v2))
        # d2Ndmdu(u, m; v2, v1, v*)
        _check_second_derivative(d2Fdudm, (1, 1), (vstar_N, v1, v2))

        # action(...)
        # d2Ndu(u, m; w, v1, v*)
        _check_second_derivative(action(d2Fdu, w), (2, 0), (vstar_N, v1,), (vstar_N, v1, w))
        # d2Ndm(u, m; p, v1, v*)
        _check_second_derivative(action(d2Fdm, p), (0, 2), (vstar_N, v1,), (vstar_N, v1, p))
        # d2Ndmdu(u, m; w, v1, v*)
        _check_second_derivative(action(d2Fdmdu, w), (1, 1), (vstar_N, v1,), (vstar_N, v1, w))
        # d2Ndudm(u, m; p, v1, v*)
        _check_second_derivative(action(d2Fdudm, p), (1, 1), (vstar_N, v1,), (vstar_N, v1, p))

        # adjoint(action(...))
        # d2Ndu(u, m; v*, v1, w)
        _check_second_derivative(adjoint(action(d2Fdu, w)), (2, 0), (vstar_N, v1), (w, v1, vstar_N))
        # d2Ndm(u, m; v*, v1, p)
        _check_second_derivative(adjoint(action(d2Fdm, p)), (0, 2), (vstar_N, v1), (p, v1, vstar_N))
        # d2Ndmdu(u, m; v*, v1, w)
        _check_second_derivative(adjoint(action(d2Fdmdu, w)), (1, 1), (vstar_N, v1), (w, v1, vstar_N))
        # d2Ndudm(u, m; v*, v1, p)
        _check_second_derivative(adjoint(action(d2Fdudm, p)), (1, 1), (vstar_N, v1), (p, v1, vstar_N))

        # action(adjoint(action(...)), ...)
        # d2Ndu(u, m; q, v1, w)
        v1 = Argument(V, 0)  # Need to renumber v1 since the argument counting starts from 0
        _check_second_derivative(action(adjoint(action(d2Fdu, w)), q), (2, 0), (v1,), (w, v1, q))
        # d2Ndm(u, m; q, v1, p)
        _check_second_derivative(action(adjoint(action(d2Fdm, p)), q), (0, 2), (v1,), (p, v1, q))
        # d2Ndmdu(u, m; q, v1, w)
        _check_second_derivative(action(adjoint(action(d2Fdmdu, w)), q), (1, 1), (v1,), (w, v1, q))
        # d2Ndudm(u, m; q, v1, p)
        _check_second_derivative(action(adjoint(action(d2Fdudm, p)), q), (1, 1), (v1,), (p, v1, q))
"""


def test_grad():

    V = FiniteElement("CG", triangle, 1)
    u_test = Coefficient(V)
    u = Coefficient(V)

    # Define an external operator equipped with a _grad method
    class ExternalOperatorCustomGrad(ExternalOperator):
        """An external operator class implementing its own spatial derivatives"""
        def __init__(self, *args, u_test, **kwargs):
            ExternalOperator.__init__(self, *args, **kwargs)
            self.u_test = u_test

        def _ufl_expr_reconstruct_(self, *args, **kwargs):
            r"""Overwrite _ufl_expr_reconstruct_ in order to keep the information
                about u_test when the operator is reconstructed.
            """
            kwargs['add_kwargs'] = {'u_test': self.u_test}
            return ExternalOperator._ufl_expr_reconstruct_(self, *args, **kwargs)

        def grad(self):
            r"""Trivial grad implementation that returns the gradient of a given
                coefficient u_test in order to check that this implementation
                is taken into account during form compiling.
            """
            return grad(self.u_test)

    # External operator with no specific gradient implementation provided
    #  -> Differentiation rules will turn grad(e) into grad(e.result_coefficient())
    # where e.result_coefficient() is the Coefficient produced by e
    e = ExternalOperator(u, function_space=V)
    expr = grad(e)
    assert expr != grad(e.result_coefficient())

    expr = expand_derivatives(expr)
    assert expr == grad(e.result_coefficient())

    # External operator with a specific gradient implementation provided
    #  -> Differentiation rules will call e_cg._grad() and use the output to replace grad(e)
    e_cg = ExternalOperatorCustomGrad(u, u_test=u_test, function_space=V)
    expr = grad(e_cg)
    assert expr != grad(u_test)

    expr = expand_derivatives(expr)
    assert expr == grad(u_test)


def test_multiple_external_operators():

    V = FiniteElement("CG", triangle, 1)
    W = FiniteElement("CG", triangle, 2)
    u = Coefficient(V)
    m = Coefficient(V)
    w = Coefficient(W)

    v = TestFunction(V)
    v_hat = TrialFunction(V)
    w_hat = TrialFunction(W)

    # N1(u, m; v*)
    N1 = ExternalOperator(u, m, function_space=V)

    # N2(w; v*)
    N2 = ExternalOperator(w, function_space=W)

    # N3(u; v*)
    N3 = ExternalOperator(u, function_space=V)

    # N4(N1, u; v*)
    N4 = ExternalOperator(N1, u, function_space=V)

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

    # dFdu = \partial F/\partial u + Action(\partial F/\partial N1, dN1/du) + Action(\partial F/\partial N4, dN4/du)
    #      = Action(\partial F/\partial N4, dN4/du), since \partial F/\partial u = 0 and \partial F/\partial N1 = 0
    #
    # In addition, we have:
    # dN4/du = \partial N4/\partial u + Action(\partial N4/\partial N1, dN1/du)
    #
    # Using the fact that Action is distributive, we have:
    #
    # dFdu = Action(\partial F/\partial N4, \partial N4/\partial u) +
    #         Action(\partial F/\partial N4, Action(\partial N4/\partial N1, dN1/du))
    dFdu = expand_derivatives(derivative(F, u))
    dFdN4_partial = inner(v_hat, v) * dx
    dN4dN1_partial = N4._ufl_expr_reconstruct_(N1, u, derivatives=(1, 0), argument_slots=N4.arguments() + (v_hat,))
    dN4du_partial = N4._ufl_expr_reconstruct_(N1, u, derivatives=(0, 1), argument_slots=N4.arguments() + (v_hat,))

    assert dFdu == Action(dFdN4_partial, Action(dN4dN1_partial, dN1du)) + Action(dFdN4_partial, dN4du_partial)

    # dFdm = Action(\partial F/\partial N4, Action(\partial N4/\partial N1, dN1/dm))
    dFdm = expand_derivatives(derivative(F, m))

    assert dFdm == Action(dFdN4_partial, Action(dN4dN1_partial, dN1dm))

    # --- F = < N1(u, m; v*), v > + <N2(w; v*), v> + <N3(u; v*), v> + < N4(N1(u, m), u; v*), v > --- #

    F = (inner(N1, v) + inner(N2, v) + inner(N3, v) + inner(N4, v)) * dx

    dFdu = expand_derivatives(derivative(F, u))
    assert dFdu == Action(dFdN1, dN1du) + Action(dFdN3, dN3du) +\
                   Action(dFdN4_partial, Action(dN4dN1_partial, dN1du)) +\
                   Action(dFdN4_partial, dN4du_partial)

    dFdm = expand_derivatives(derivative(F, m))
    assert dFdm == Action(dFdN1, dN1dm) + Action(dFdN4_partial, Action(dN4dN1_partial, dN1dm))

    dFdw = expand_derivatives(derivative(F, w))
    assert dFdw == Action(dFdN2, dN2dw)
