#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

__authors__ = "Nacime Bouziani"
__date__ = "2019-03-26"


"""
Test ExternalOperator object
"""

import pytest
import math 

# This imports everything external code will see from ufl
from ufl import *
from ufl.core.external_operator import ExternalOperator
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.constantvalue import as_ufl
from ufl.domain import default_domain


def test_properties(self, cell):
    S = FiniteElement("CG", cell, 1)
    cs = Constant(cell)
    u = Coefficient(S)
    v = Coefficient(S)
    r = Coefficient(S)

    nl = ExternalOperator(u, r, eval_space=S)

    domain = default_domain(cell)
    space = FunctionSpace(domain, S)

    assert nl.ufl_function_space() == space
    assert nl.ufl_operands[0] == u
    assert nl.ufl_operands[1] == r
    assert nl.derivatives == (0, 0)
    assert nl.ufl_shape == ()

    nl2 = ExternalOperator(u, r, eval_space=S, derivatives=(3, 4), shape = (2,))
    assert nl2.derivatives == (3, 4)
    assert nl2.ufl_shape == (2,)

def _test(f, df):
    v =variable(5.0)
    v1 = variable(6.0)
    P = FiniteElement("Quadrature", triangle, 2)
    dfvtest = diff(f(v, P), v1)
    
    dfv1 = diff(f(v, P), v)
    dfv2 = df(v, P)
    assert apply_derivatives(dfv1) == dfv2 

def _test_multivariable(f, df1, df2, df3):
    v1 = variable(4450.567)
    v2 = variable(3495.348)
    v3 = variable(1294.387)
    P = FiniteElement("Quadrature", triangle, 2)

    dfv1 = diff(f(v1, v2, v3, P), v1)
    dfv2 = df1(v1, v2, v3, P)
    assert apply_derivatives(dfv1) == dfv2

    dfv1 = diff(f(v1, v2, v3, P), v2)
    dfv2 = df2(v1, v2, v3, P)
    assert apply_derivatives(dfv1) == dfv2

    dfv1 = diff(f(v1, v2, v3, P), v3)
    dfv2 = df3(v1, v2, v3, P)
    assert apply_derivatives(dfv1) == dfv2

def testVariable():
    def f(v, space):
        return ExternalOperator(v, eval_space=space)

    def df(v, space):
        e = f(v, space)
        nl = e._ufl_expr_reconstruct_(v, derivatives=(1,), eval_space=space, count=e._count - 1)
        return as_ufl(nl)

    def df2(v, space):
        e = f(v, space)
        nl = e._ufl_expr_reconstruct_(v, derivatives=(2,), eval_space=space, count=e._count - 2)
        return as_ufl(nl)
    _test(f, df)
    _test(df, df2)

def testProduct():
    def g(v, space):
        nl = ExternalOperator(v, eval_space=space)
        return nl

    def f(v,space):
        return 3*g(v,space)

    def df(v, space):
        e = g(v, space)
        nl = e._ufl_expr_reconstruct_(v, derivatives=(1,), eval_space=space, count=e._count-1)
        return as_ufl(3*nl)
    _test(f, df)

def testProductExternalOperator():
    cst = 2.0
    def g(v, space):
        nl = ExternalOperator(cst*v, eval_space=space)
        nl2 = ExternalOperator(v, derivatives=(1,), eval_space=space)
        return nl, nl2

    def f(v, space):
        gg = g(v,space)
        nl = gg[0]
        nl2 = gg[1]
        return nl*nl2

    def df(v, space):
        gg = g(v, space)
        e1 = gg[0]
        e2 = gg[1]

        nl = e1._ufl_expr_reconstruct_(cst*v, eval_space=space, count=e1._count-2)
        nl2 = e2._ufl_expr_reconstruct_(v,derivatives=(1,), eval_space=space, count=e2._count-2)
        dnl = cst*e1._ufl_expr_reconstruct_(cst*v,derivatives=(1,), eval_space=space, count=e1._count-2)
        dnl2 = e2._ufl_expr_reconstruct_(v,derivatives=(2,), eval_space=space, count=e2._count-2)

        return as_ufl(dnl*nl2+dnl2*nl)
    _test(f, df)

def testmultiVariable():
    def g(v1, v2, v3, space):
        return ExternalOperator(v1, v2, v3, eval_space=space)
    def f(v1, v2, v3, space):
        return cos(v1)*sin(v2)*g(v1, v2, v3, space)
    def df1(v1, v2, v3, space):
        r = g(v1, v2, v3, space)
        g1 = r._ufl_expr_reconstruct_(v1, v2, v3, derivatives=(0, 0, 0), eval_space=space, count=r._count-1)
        g2 = r._ufl_expr_reconstruct_(v1, v2, v3, derivatives=(1, 0, 0), eval_space=space, count=r._count-1)
        nl =  - sin(v1)*sin(v2)*g1 + cos(v1)*sin(v2)*g2
        return as_ufl(nl)
    def df2(v1, v2, v3, space):
        r = g(v1,v2,v3,space)
        g1 = r._ufl_expr_reconstruct_(v1, v2, v3,derivatives=(0, 0, 0), eval_space=space, count=r._count-1)
        g2 = r._ufl_expr_reconstruct_(v1, v2, v3,derivatives=(0, 1, 0), eval_space=space, count=r._count-1)
        nl = cos(v2)*cos(v1)*g1 + cos(v1)*sin(v2)*g2
        return as_ufl(nl)
    def df3(v1, v2, v3, space):
        r = g(v1, v2, v3, space)
        g1 = r._ufl_expr_reconstruct_(v1, v2, v3, derivatives=(0, 0, 1), eval_space=space, count=r._count-1)
        nl = cos(v1)*sin(v2)*g1
        return as_ufl(nl)
    _test_multivariable(f, df1, df2, df3)

def test_form(self):
    cell = triangle
    V = FiniteElement("CG", cell, 1)
    P = FiniteElement("Quadrature", cell, 2)
    u = Coefficient(V)
    m = Coefficient(V)
    u_hat = Coefficient(V)
    v = TestFunction(V)

    nl = ExternalOperator(u, m, eval_space=P)
    a = nl * v
    actual = derivative(a, u, u_hat)

    dnl_du = nl
    dnl_du = nl._ufl_expr_reconstruct_(u, m, derivatives=(1, 0), eval_space=P, count=nl._count) 
    expected = u_hat*dnl_du*v
    assert apply_derivatives(actual) == expected
