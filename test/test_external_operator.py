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
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms import expand_derivatives
from ufl.constantvalue import as_ufl
from ufl.domain import default_domain

from functools import partial


def test_properties(self, cell):
    S = FiniteElement("CG", cell, 1)
    u = Coefficient(S)
    r = Coefficient(S)

    nl = ExternalOperator(u, r, function_space=S)

    domain = default_domain(cell)
    space = FunctionSpace(domain, S)

    assert nl.ufl_function_space() == space
    assert nl.ufl_operands[0] == u
    assert nl.ufl_operands[1] == r
    assert nl.derivatives == (0, 0)
    assert nl.ufl_shape == ()

    nl2 = ExternalOperator(u, r, function_space=S, derivatives=(3, 4))
    assert nl2.derivatives == (3, 4)
    assert nl2.ufl_shape == ()


def _create_external_operator(V=None, nops=1):
    if V is None:
        space = FiniteElement("Quadrature", triangle, 1)
    else:
        space = V
    return ExternalOperator(*[variable(0.) for i in range(nops)], function_space=space)


def _test(f, df):
    v = variable(5.0)
    v1 = variable(6.0)
    fexpr = f(v)

    dfv1 = diff(fexpr, v)
    dfv2 = df(v)
    assert apply_derivatives(dfv1) == dfv2


def _test_multivariable(f, df1, df2, df3):
    v1 = variable(4450.567)
    v2 = variable(3495.348)
    v3 = variable(1294.387)
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


def testVariable():
    V = FiniteElement("Quadrature", triangle, 2)
    e = _create_external_operator(V)

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
    _test(fe, dfe)
    _test(dfe, df2e)


def testProduct():
    V = FiniteElement("Quadrature", triangle, 3)
    e = _create_external_operator(V)

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
    _test(fe, dfe)


def testProductExternalOperator():
    V = FiniteElement("Quadrature", triangle, 3)
    e1 = _create_external_operator(V)
    e2 = _create_external_operator(V)

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
    _test(fe, dfe)


def testmultiVariable():
    V = FiniteElement("Quadrature", triangle, 3)
    e = _create_external_operator(V, 3)

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
    _test_multivariable(fe, df1e, df2e, df3e)


def test_form():
    cell = triangle
    V = FiniteElement("CG", cell, 1)
    P = FiniteElement("Quadrature", cell, 2)
    u = Coefficient(V)
    m = Coefficient(V)
    u_hat = Coefficient(V)
    v = TestFunction(V)

    nl = ExternalOperator(u, m, function_space=P)
    a = nl * v
    actual = derivative(a, u, u_hat)

    dnl_du = nl
    dnl_du = nl._ufl_expr_reconstruct_(u, m, derivatives=(1, 0))
    expected = u_hat * dnl_du * v
    assert apply_derivatives(actual) == expected


def test_dependency():
    V = FiniteElement("CG", triangle, 1)
    Vv = VectorElement("CG", triangle, 1)

    u = Coefficient(V)
    w = Coefficient(V)

    e = ExternalOperator(u, w, function_space=V)

    dedu = e._ufl_expr_reconstruct_(u, w, derivatives=(1, 0))
    dedw = e._ufl_expr_reconstruct_(u, w, derivatives=(0, 1))

    d2edu = dedu._ufl_expr_reconstruct_(u, w, derivatives=(2, 0))

    assert e == dedu._extop_master
    assert e == dedw._extop_master

    assert e._extop_dependencies[0] == e
    assert e._extop_dependencies[1] == dedu
    assert e._extop_dependencies[2] == dedw

    assert e._extop_dependencies[3] == d2edu
    assert len(e._extop_dependencies) == 4

    e2 = ExternalOperator(u, w, grad(u), div(w), function_space=V)
    der = [(0, 0, 0, 1), (1, 0, 0, 1), (2, 0, 1, 1)]
    args = [(), (), ()]
    e2._add_dependencies(der, args)

    assert e2._extop_dependencies[0] == e2
    assert e2._extop_dependencies[1].derivatives == der[0]
    assert e2._extop_dependencies[2].derivatives == der[1]
    assert e2._extop_dependencies[3].derivatives == der[2]
    assert e2._extop_dependencies[1]._extop_master == e2
    assert e2._extop_dependencies[2]._extop_master == e2
    assert e2._extop_dependencies[3]._extop_master == e2

    e3 = ExternalOperator(u, function_space=V)
    u_hat = Coefficient(V)

    a = inner(grad(e3), grad(w))
    Ja = derivative(a, u, u_hat)
    expand_derivatives(Ja)

    assert len(e3._extop_dependencies) == 3
    assert e3._extop_dependencies[0].derivatives == (0,)
    assert e3._extop_dependencies[1].derivatives == (1,)
    assert e3._extop_dependencies[2].derivatives == (2,)
    assert e3._extop_dependencies[0]._extop_master == e3
    assert e3._extop_dependencies[1]._extop_master == e3
    assert e3._extop_dependencies[2]._extop_master == e3

    e4 = ExternalOperator(u, function_space=Vv)

    a = inner(div(e4), w)
    Ja = derivative(a, u, u_hat)
    expand_derivatives(Ja)

    assert len(e4._extop_dependencies) == 3
    assert e4._extop_dependencies[0].derivatives == (0,)
    assert e4._extop_dependencies[1].derivatives == (1,)
    assert e4._extop_dependencies[2].derivatives == (2,)
    assert e4._extop_dependencies[0]._extop_master == e4
    assert e4._extop_dependencies[1]._extop_master == e4
    assert e4._extop_dependencies[2]._extop_master == e4


def test_function_spaces_derivatives():
    V = FiniteElement("CG", triangle, 1)
    Vv = VectorElement("CG", triangle, 1)
    Vt = TensorElement("CG", triangle, 1)
    Vt2 = TensorElement(V, shape=(2, 2, 2))
    Vt3 = TensorElement(V, shape=(2, 2, 2, 2))
    Vt4 = TensorElement(V, shape=(2, 2, 2, 2, 2))
    Vt5 = TensorElement(V, shape=(2, 2, 2, 2, 2, 2))

    def _check_space_shape_fct_space(x, der, shape, space, original_space):
        assert x.derivatives == der
        assert x.ufl_shape == shape
        assert x.ufl_function_space().ufl_element() == space
        assert x.original_function_space().ufl_element() == original_space

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

    _check_space_shape_fct_space(dedu, (1, 0), (), V, V)
    _check_space_shape_fct_space(dedw, (0, 1), (), V, V)

    _check_space_shape_fct_space(d2edu, (2, 0), (), V, V)
    _check_space_shape_fct_space(dedwdu, (1, 1), (), V, V)
    _check_space_shape_fct_space(d2edw, (0, 2), (), V, V)

    # Vector case
    ev = ExternalOperator(uv, w, function_space=Vv)
    deduv = ev._ufl_expr_reconstruct_(uv, w, derivatives=(1, 0))
    dedw = ev._ufl_expr_reconstruct_(uv, w, derivatives=(0, 1))
    d2eduv = ev._ufl_expr_reconstruct_(uv, w, derivatives=(2, 0))
    dedwduv = ev._ufl_expr_reconstruct_(uv, w, derivatives=(1, 1))
    d2edw = ev._ufl_expr_reconstruct_(uv, w, derivatives=(0, 2))

    _check_space_shape_fct_space(deduv, (1, 0), (2, 2), Vv, Vt)
    _check_space_shape_fct_space(dedw, (0, 1), (2,), Vv, Vv)

    _check_space_shape_fct_space(d2eduv, (2, 0), (2, 2, 2), Vv, Vt2)
    _check_space_shape_fct_space(dedwduv, (1, 1), (2, 2), Vv, Vt)
    _check_space_shape_fct_space(d2edw, (0, 2), (2,), Vv, Vv)

    # Tensor case
    et = ExternalOperator(ut, uv, w, function_space=Vt)
    dedut = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(1, 0, 0))
    deduv = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 1, 0))
    dedw = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 0, 1))

    _check_space_shape_fct_space(dedut, (1, 0, 0), (2, 2, 2, 2), Vt, Vt3)
    _check_space_shape_fct_space(deduv, (0, 1, 0), (2, 2, 2), Vt, Vt2)
    _check_space_shape_fct_space(dedw, (0, 0, 1), (2, 2), Vt, Vt)

    d2edut = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(2, 0, 0))
    d2eduv = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 2, 0))
    d2edw = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 0, 2))

    _check_space_shape_fct_space(d2edut, (2, 0, 0), (2, 2, 2, 2, 2, 2), Vt, Vt5)
    _check_space_shape_fct_space(d2eduv, (0, 2, 0), (2, 2, 2, 2), Vt, Vt3)
    _check_space_shape_fct_space(d2edw, (0, 0, 2), (2, 2), Vt, Vt)

    dedwduv = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(0, 1, 1))
    dedwdut = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(1, 0, 1))
    dedutduv = et._ufl_expr_reconstruct_(ut, uv, w, derivatives=(1, 1, 0))

    _check_space_shape_fct_space(dedwduv, (0, 1, 1), (2, 2, 2), Vt, Vt2)
    _check_space_shape_fct_space(dedwdut, (1, 0, 1), (2, 2, 2, 2), Vt, Vt3)
    _check_space_shape_fct_space(dedutduv, (1, 1, 0), (2, 2, 2, 2, 2), Vt, Vt4)

    # TODO: MIXED ELEMENT


def test_differentiation_procedure_action():
    V = VectorElement("CG", triangle, 1)
    u = Coefficient(V)
    w = Coefficient(V)

    # Define a class with a GLOBAL external operator type to check the differentiation procedure for the action case
    class ActionExternalOperator(ExternalOperator):

        _external_operator_type = 'GLOBAL'

        def __init__(self, *args, **kwargs):
            ExternalOperator.__init__(self, *args, **kwargs)

    # External operators
    e_action = ActionExternalOperator(u, w, function_space=V)
    e = ExternalOperator(u, w, function_space=V)

    u_hat = TrialFunction(V)

    # Bilinear forms
    a = inner(e, w) * dx
    Ja = derivative(a, u, u_hat)
    Ja = expand_derivatives(Ja)

    a_action = inner(e_action, w) * dx
    Ja_action = derivative(a_action, u, u_hat)
    Ja_action = expand_derivatives(Ja_action)

    # Get external operators
    extop_Ja, = Ja.external_operators()
    extop_Ja_action, = Ja_action.external_operators()

    # Check derivatives
    assert extop_Ja.derivatives == (1, 0)
    assert extop_Ja_action.derivatives == (1, 0)

    # Check arguments
    assert extop_Ja.arguments() == ()
    assert extop_Ja_action.arguments() == ((u_hat, False),)
    assert extop_Ja.action_args() == ()
    assert extop_Ja_action.action_args() == ()

    # Check shape
    assert extop_Ja.ufl_shape == (2, 2)
    assert extop_Ja_action.ufl_shape == (2,)


def test_extractions():
    from ufl.algorithms.analysis import (extract_coefficients, extract_arguments,
                                         extract_arguments_and_coefficients,
                                         extract_external_operators, extract_constants)

    V = FiniteElement("CG", triangle, 1)
    u = Coefficient(V)
    c = Constant(triangle)

    e = ExternalOperator(u, c, function_space=V)

    assert extract_coefficients(e) == [u, e.coefficient]
    assert extract_arguments_and_coefficients(e) == ([], [u, e.coefficient])
    assert extract_constants(e) == [c]
    assert extract_external_operators(e) == [e]

    F = e * dx

    assert extract_coefficients(F) == [u, e.coefficient]
    assert extract_arguments_and_coefficients(e) == ([], [u, e.coefficient])
    assert extract_constants(F) == [c]
    assert F.external_operators() == (e,)

    u_hat = TrialFunction(V)
    e = ExternalOperator(u, function_space=V, arguments=((u_hat, False),))

    assert extract_coefficients(e) == [u, e.coefficient]
    assert extract_arguments(e) == [u_hat]
    assert extract_arguments_and_coefficients(e) == ([u_hat], [u, e.coefficient])
    assert extract_external_operators(e) == [e]

    F = e * dx

    assert extract_coefficients(F) == [u, e.coefficient]
    assert extract_arguments(e) == [u_hat]
    assert extract_arguments_and_coefficients(e) == ([u_hat], [u, e.coefficient])
    assert F.external_operators() == (e,)

    w = Coefficient(V)
    e2 = ExternalOperator(w, e, function_space=V)

    assert extract_coefficients(e2) == [u, e.coefficient, w, e2.coefficient]
    assert extract_arguments(e2) == [u_hat]
    assert extract_arguments_and_coefficients(e2) == ([u_hat], [u, e.coefficient, w, e2.coefficient])
    assert extract_external_operators(e2) == [e, e2]

    F = e2 * dx

    assert extract_coefficients(e2) == [u, e.coefficient, w, e2.coefficient]
    assert extract_arguments(e2) == [u_hat]
    assert extract_arguments_and_coefficients(e2) == ([u_hat], [u, e.coefficient, w, e2.coefficient])
    assert F.external_operators() == (e, e2)
