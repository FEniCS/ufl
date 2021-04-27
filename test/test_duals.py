#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

from ufl import *

__authors__ = "India Marsden"
__date__ = "2020-12-28 -- 2020-12-28"

import pytest

from ufl.domain import default_domain
from ufl.duals import is_primal,is_dual



def test_mixed_functionspace(self):
    # Domains
    domain_3d = default_domain(tetrahedron)
    domain_2d = default_domain(triangle)
    domain_1d = default_domain(interval)
    # Finite elements
    f_1d = FiniteElement("CG", interval, 1)
    f_2d = FiniteElement("CG", triangle, 1)
    f_3d = FiniteElement("CG", tetrahedron, 1)
    # Function spaces
    V_3d = FunctionSpace(domain_3d, f_3d)
    V_2d = FunctionSpace(domain_2d, f_2d)
    V_1d = FunctionSpace(domain_1d, f_1d)

    # MixedFunctionSpace = V_3d x V_2d x V_1d
    V = MixedFunctionSpace(V_3d, V_2d, V_1d)
    # Check sub spaces
    assert(is_primal(V_3d))
    assert(is_primal(V_2d))
    assert(is_primal(V_1d))
    assert(is_primal(V))

     # Get dual of V_3
    V_dual = V_3d.dual()

    #  Test dual functions on MixedFunctionSpace = V_dual x V_2d x V_1d
    V = MixedFunctionSpace(V_dual, V_2d, V_1d)
    V_mixed_dual = MixedFunctionSpace(V_dual, V_2d.dual(), V_1d.dual())

    assert(is_dual(V_dual))
    assert(not is_dual(V))
    assert(is_dual(V_mixed_dual))

def test_dual_coefficients():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    v = Coefficient(V, count=1)
    u = Coefficient(V_dual, count=1)
    w = Cofunction(V_dual)
    x = Cofunction(V)

    assert(is_primal(v))
    assert(not is_dual(v))

    assert(is_dual(u))
    assert(not is_primal(u))

    assert(is_dual(w))
    assert(not is_primal(w))

    assert(is_primal(x))
    assert(not is_dual(x))

def test_dual_arguments():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    v = Argument(V,1)
    u = Argument(V_dual, 2)
    w = Coargument(V_dual, 3)
    x = Coargument(V, 4)

    assert(is_primal(v))
    assert(not is_dual(v))

    assert(is_dual(u))
    assert(not is_primal(u))

    assert(is_dual(w))
    assert(not is_primal(w))

    assert(is_primal(x))
    assert(not is_dual(x))

def test_addition():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    u = TrialFunction(V)
    v = TestFunction(V)

    # linear 1-form
    L = v * dx
    a = Cofunction(V_dual)
    res = L + a
    assert(isinstance(res,FormSum))
    assert(res)

    L = u * v * dx
    a = Matrix(V, V)
    res = L + a
    assert(isinstance(res,FormSum))
    assert(res)

def test_scalar_mult():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    # linear 1-form
    a = Cofunction(V_dual)
    res = 2 * a
    assert(isinstance(res,FormSum))
    assert(res)

    a = Matrix(V, V)
    res = 2 * a
    assert(isinstance(res,FormSum))
    assert(res)

def test_adjoint():
    V = FiniteElement("CG", triangle, 1)
    a = Matrix(V, V)

    adjoint = Adjoint(a)
    res = 2 * adjoint
    assert(isinstance(res,FormSum))
    assert(res)

    res = Adjoint(2 * a)
    assert(isinstance(res,FormSum))
    assert(isinstance(res.components()[0], Adjoint))