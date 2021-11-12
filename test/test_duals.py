#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

from ufl import FiniteElement, FunctionSpace, MixedFunctionSpace, \
    Coefficient, Matrix, Cofunction, FormSum, Argument, Coargument,\
    TestFunction, TrialFunction, Adjoint, Action, \
    action, adjoint, tetrahedron, triangle, interval, dx

__authors__ = "India Marsden"
__date__ = "2020-12-28 -- 2020-12-28"

import pytest

from ufl.domain import default_domain
from ufl.duals import is_primal, is_dual
# from ufl.algorithms.ad import expand_derivatives


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
    assert is_primal(V_3d)
    assert is_primal(V_2d)
    assert is_primal(V_1d)
    assert is_primal(V)

    # Get dual of V_3
    V_dual = V_3d.dual()

    #  Test dual functions on MixedFunctionSpace = V_dual x V_2d x V_1d
    V = MixedFunctionSpace(V_dual, V_2d, V_1d)
    V_mixed_dual = MixedFunctionSpace(V_dual, V_2d.dual(), V_1d.dual())

    assert is_dual(V_dual)
    assert not is_dual(V)
    assert is_dual(V_mixed_dual)


def test_dual_coefficients():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    v = Coefficient(V, count=1)
    u = Coefficient(V_dual, count=1)
    w = Cofunction(V_dual)

    assert is_primal(v)
    assert not is_dual(v)

    assert is_dual(u)
    assert not is_primal(u)

    assert is_dual(w)
    assert not is_primal(w)

    try:
        x = Cofunction(V)
        assert False
    except ValueError:
        pass


def test_dual_arguments():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    v = Argument(V, 1)
    u = Argument(V_dual, 2)
    w = Coargument(V_dual, 3)

    assert is_primal(v)
    assert not is_dual(v)

    assert is_dual(u)
    assert not is_primal(u)

    assert is_dual(w)
    assert not is_primal(w)

    try:
        x = Coargument(V, 4)
        assert False
    except ValueError:
        pass


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
    assert isinstance(res, FormSum)
    assert res

    L = u * v * dx
    a = Matrix(V, V)
    res = L + a
    assert isinstance(res, FormSum)
    assert res


def test_scalar_mult():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    # linear 1-form
    a = Cofunction(V_dual)
    res = 2 * a
    assert isinstance(res, FormSum)
    assert res

    a = Matrix(V, V)
    res = 2 * a
    assert isinstance(res, FormSum)
    assert res


def test_adjoint():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    a = Matrix(V, V)

    adj = adjoint(a)
    res = 2 * adj
    assert isinstance(res, FormSum)
    assert res

    res = adjoint(2 * a)
    assert isinstance(res, FormSum)
    assert isinstance(res.components()[0], Adjoint)


def test_action():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    domain_1d = default_domain(interval)
    f_1d = FiniteElement("CG", interval, 1)
    U = FunctionSpace(domain_1d, f_1d)

    a = Matrix(V, U)
    b = Matrix(V, U.dual())
    u = Coefficient(U)
    u_a = Argument(U, 0)
    v = Coefficient(V)
    u_star = Cofunction(U.dual())
    u_form = u_a * dx

    res = action(a, u)
    assert res
    assert len(res.arguments()) < len(a.arguments())
    assert isinstance(res, Action)

    repeat = action(res, v)
    assert repeat
    assert len(repeat.arguments()) < len(res.arguments())

    res = action(2 * a, u)
    assert isinstance(res, FormSum)
    assert isinstance(res.components()[0], Action)

    res = action(b, u_form)
    assert res
    assert len(res.arguments()) < len(b.arguments())

    with pytest.raises(TypeError):
        res = action(a, v)

    with pytest.raises(TypeError):
        res = action(a, u_star)


"""
def test_differentiation():
    domain_2d = default_domain(triangle)
    f_2d = FiniteElement("CG", triangle, 1)
    V = FunctionSpace(domain_2d, f_2d)
    domain_1d = default_domain(interval)
    f_1d = FiniteElement("CG", interval, 1)
    U = FunctionSpace(domain_1d, f_1d)

    u = Coefficient(U)
    # Matrix
    M = Matrix(V, U)
    # Cofunction
    u_star = Cofunction(U.dual())
    # Action
    Ac = Action(M, u)
    # Adjoint
    Ad = Adjoint(M)
    # Form sum
    Fs = M + Ad

    dMdu = expand_derivatives(derivative(M, u))
"""
