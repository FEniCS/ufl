#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

from ufl import FiniteElement, FunctionSpace, Coefficient, Argument, triangle

__authors__ = "Nacime Bouziani"
__date__ = "2021-11-19"

"""
Test Interp object
"""

from ufl.core.interp import Interp
from ufl.domain import default_domain


def test_symbolic():

    # -- Set function spaces -- #
    domain_2d = default_domain(triangle)
    f1 = FiniteElement("CG", triangle, 1)
    V1 = FunctionSpace(domain_2d, f1)

    f2 = FiniteElement("CG", triangle, 2)
    V2 = FunctionSpace(domain_2d, f2)
    V2_dual = V2.dual()

    u = Coefficient(V1)
    vstar = Argument(V2_dual, 0)
    Iu = Interp(u, vstar)

    assert Iu == Interp(u, V2)
    assert Iu.ufl_function_space() == V2
    assert Iu.argument_slots() == (vstar, u)
    assert Iu.arguments() == (vstar,)
    assert Iu.ufl_operands == ()

    # Check trivial cases
    I = Interp(u, V1)
    assert I == u

    I = Interp(0, V1)
    assert I == 0


def test_action_adjoint():

    # -- Set function spaces -- #
    domain_2d = default_domain(triangle)
    f1 = FiniteElement("CG", triangle, 1)
    V1 = FunctionSpace(domain_2d, f1)

    f2 = FiniteElement("CG", triangle, 2)
    V2 = FunctionSpace(domain_2d, f2)
    V2_dual = V2.dual()

    # u = Coefficient(V1)
    vstar = Argument(V2_dual, 0)
    # Iu = Interp(u, vstar)

    v = Argument(V1, 1)
    Iv = Interp(v, vstar)

    assert Iv.argument_slots() == (vstar, v)
    assert Iv.arguments() == (vstar, v)

    # -- Add tests for Action/Adjoint -- #


"""
def test_differentiation():

    # Put differentiation equivalence: dInterp(u, v*)/du[uhat] <==> Interp(uhat, v*)
"""
