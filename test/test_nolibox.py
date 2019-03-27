#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

__authors__ = "Nacime Bouziani"
__date__ = "2019-03-26"


"""
Test Nolibox object
"""

import pytest
import math 

# This imports everything external code will see from ufl
from ufl import *
from ufl.core.nolibox import Nolibox
from ufl.algebra import Sum
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.constantvalue import as_ufl
from ufl.algorithms import expand_derivatives

from test_derivative import assertEqualBySampling

def get_variables():
    xv = None
    vv = 5.0
    return (xv, variable(vv))

@pytest.fixture
def v():
    xv, vv = get_variables()
    return vv


def test_properties(self, cell):
   
    S = FiniteElement("CG", cell, 1)
    cs = Constant(cell)
    u = Coefficient(S)
    v = Coefficient(S)
    r = Coefficient(S)
    
    nl = Nolibox(u,r,eval_space=S)
    
    space = S
    num_op = 2

    assert nl.eval_space == space
    assert nl.ufl_operands[0] == u
    assert nl.ufl_operands[1] == r
    assert nl.deriv_index == (0,0)

    nl2 = Nolibox(u,r,eval_space=S, derivatives = (3,4))
    assert nl2.deriv_index == (3,4)
    
    
    
def _test(f, df):
    x, v = get_variables()

    dfv1 = diff(f(v), v)
    dfv2 = df(v)
    dfv1 = dfv1(x)
    dfv2 = dfv2(x)
    assert round(dfv1 - dfv2, 7) == 0

    dfv1 = diff(f(7 * v), v)
    dfv2 = 7 * df(7 * v)
    dfv1 = dfv1(x)
    dfv2 = dfv2(x)
    assert round(dfv1 - dfv2, 7) == 0

def _test_multivariable(f, df1, df2, df3):
    x = None
    v1 = variable(4450.567)
    v2 = variable(3495.348)
    v3 = variable(1294.387)

    dfv1 = diff(f(v1,v2,v3), v1)
    dfv2 = df1(v1,v2,v3)
    dfv1 = dfv1(x)
    dfv2 = dfv2(x)
    assert round(dfv1 - dfv2, 7) == 0
    
    dfv1 = diff(f(v1,v2,v3), v2)
    dfv2 = df2(v1,v2,v3)
    dfv1 = dfv1(x)
    dfv2 = dfv2(x)
    assert round(dfv1 - dfv2, 7) == 0
    
    dfv1 = diff(f(v1,v2,v3), v3)
    dfv2 = df3(v1,v2,v3)
    dfv1 = dfv1(x)
    dfv2 = dfv2(x)
    assert round(dfv1 - dfv2, 7) == 0
   
    
def testVariable(v):
    def f(v):
        nl = Nolibox(v)
        return nl

    def df(v):
        nl = Nolibox(v,derivatives=(1,))
        return as_ufl(nl)
    
    def d2f(v):
        nl = Nolibox(v,derivatives=(2,))
        return as_ufl(nl)
    _test(f, df)
    _test(df,d2f)


def testProduct(v):
    def f(v):
        nl = Nolibox(v)
        return 3 * nl

    def df(v):
        nl = Nolibox(v,derivatives=(1,))
        return as_ufl(3*nl)
    _test(f, df)
    
def testProductNolibox(v):
    def f(v):
        nl = Nolibox(2*v)
        nl2 = Nolibox(v,derivatives=(1,))
        return nl*nl2

    def df(v):
        nl = Nolibox(2*v)
        nl2 = Nolibox(v,derivatives=(1,))
        dnl = 2*Nolibox(2*v,derivatives=(1,))
        dnl2 = Nolibox(v,derivatives=(2,))
        return as_ufl(dnl*nl2+dnl2*nl)
    _test(f, df)
    
def testmultiVariable(v):
    def f(v1,v2,v3):
        nl = cos(v1)*sin(v2)*Nolibox(v1,v2,v3)
        return nl
    def df1(v1,v2,v3):
        nl = sin(v2)*(Nolibox(v1,v2,v3,derivatives=(1,0,0))*cos(v1) - sin(v1)*Nolibox(v1,v2,v3,derivatives=(0,0,0)))
        return as_ufl(nl)
    def df2(v1,v2,v3):
        nl = cos(v1)*(sin(v2)*Nolibox(v1,v2,v3,derivatives=(0,1,0)) + cos(v2)*Nolibox(v1,v2,v3,derivatives=(0,0,0)))
        return as_ufl(nl)
    def df3(v1,v2,v3):
        nl = cos(v1)*sin(v2)*Nolibox(v1,v2,v3,derivatives=(0,0,1))
        return as_ufl(nl)
    _test_multivariable(f, df1, df2, df3)
    
def test_form(self):
    #FIXME : fix the warning (Missing degree estimation handler for type Nolibox)
    cell = triangle
    V = FiniteElement("CG", cell, 1)
    u = Coefficient(V)
    m = Coefficient(V)
    u_hat = Coefficient(V)
    v = TestFunction(V)
    
    cst = 100
    nl = Nolibox(u,m)
    dnl_du = Nolibox(u,m,derivatives=(1,0))
    a = cst * nl * v
    actual = derivative(a, u, u_hat)
    expected = cst*(u_hat*dnl_du*v)
    assertEqualBySampling(actual, expected)
    
 