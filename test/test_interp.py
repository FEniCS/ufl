#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

import pytest
from ufl import FiniteElement, FunctionSpace, Coefficient, Argument, triangle, derivative

__authors__ = "Nacime Bouziani"
__date__ = "2021-11-19"

"""
Test Interp object
"""

from ufl.algorithms.ad import expand_derivatives
from ufl.core.interp import Interp
from ufl.domain import default_domain


@pytest.fixture
def V1():
    domain_2d = default_domain(triangle)
    f1 = FiniteElement("CG", triangle, 1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V2():
    domain_2d = default_domain(triangle)
    f1 = FiniteElement("CG", triangle, 2)
    return FunctionSpace(domain_2d, f1)


def test_symbolic(V1, V2):

    # Set dual of V2
    V2_dual = V2.dual()

    u = Coefficient(V1)
    vstar = Argument(V2_dual, 0)
    Iu = Interp(u, vstar)

    assert Iu == Interp(u, V2)
    assert Iu.ufl_function_space() == V2
    assert Iu.argument_slots() == (vstar, u)
    assert Iu.arguments() == (vstar,)
    assert Iu.ufl_operands == ()


def test_action_adjoint(V1, V2):

    # Set dual of V2
    V2_dual = V2.dual()

    # u = Coefficient(V1)
    vstar = Argument(V2_dual, 0)
    # Iu = Interp(u, vstar)

    v = Argument(V1, 1)
    Iv = Interp(v, vstar)

    assert Iv.argument_slots() == (vstar, v)
    assert Iv.arguments() == (vstar, v)

    # -- Add tests for Action/Adjoint -- #


def test_differentiation(V1, V2):

    u = Coefficient(V1)
    Iu = Interp(u, V2)

    v1 = Argument(V1, 1)
    dIu = expand_derivatives(derivative(Iu, u, v1))

    # dInterp(u, v*)/du[uhat] <==> Interp(uhat, v*)
    assert dIu == Interp(v1, V2)

    g = u**2
    Ig = Interp(g, V2)
    dIg = expand_derivatives(derivative(Ig, u, v1))
    assert dIg == Interp(2 * v1 * u, V2)
