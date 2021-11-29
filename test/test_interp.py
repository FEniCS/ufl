#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

import pytest
from ufl import (FiniteElement, FunctionSpace, Coefficient, Argument, triangle, derivative,
                 TestFunction, TrialFunction, action, adjoint, Action, Adjoint, dx)


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
    assert Iu.ufl_operands == (u,)


def test_action_adjoint(V1, V2):

    # Set dual of V2
    V2_dual = V2.dual()
    vstar = Argument(V2_dual, 0)

    u = Coefficient(V1)
    Iu = Interp(u, vstar)

    v1 = TrialFunction(V1)
    Iv = Interp(v1, vstar)

    assert Iv.argument_slots() == (vstar, v1)
    assert Iv.arguments() == (vstar, v1)

    # -- Action -- #
    v = TestFunction(V1)
    v2 = TrialFunction(V2)
    F = v2 * v * dx
    assert action(Iv, u) == Action(Iv, u)
    assert action(F, Iv) == Action(F, Iv)
    assert action(F, Iu) == Iu * v * dx

    # -- Adjoint -- #
    adjoint(Iv) == Adjoint(Iv)


def test_differentiation(V1, V2):

    u = Coefficient(V1)
    v = TestFunction(V1)

    # Define Interp
    Iu = Interp(u, V2)

    # -- Differentiate: Interp(u, V2) -- #
    uhat = TrialFunction(V1)
    dIu = expand_derivatives(derivative(Iu, u, uhat))

    # dInterp(u, v*)/du[uhat] <==> Interp(uhat, v*)
    assert dIu == Interp(uhat, V2)

    # -- Differentiate: Interp(u**2, V2) -- #
    g = u**2
    Ig = Interp(g, V2)
    dIg = expand_derivatives(derivative(Ig, u, uhat))
    assert dIg == Interp(2 * uhat * u, V2)

    # -- Differentiate: I(u, V2) * v * dx -- #
    F = Iu * v * dx
    Ihat = TrialFunction(Iu.ufl_function_space())
    dFdu = expand_derivatives(derivative(F, u, uhat))
    # Compute dFdu = \partial F/\partial u + Action(dFdIu, dIu/du)
    #              = Action(dFdIu, Iu(uhat, v*))
    dFdIu = Ihat * v * dx
    assert dFdu == Action(dFdIu, dIu)

    # -- Differentiate: u * I(u, V2) * v * dx -- #
    F = u * Iu * v * dx
    dFdu = expand_derivatives(derivative(F, u, uhat))
    # Compute dFdu = \partial F/\partial u + Action(dFdIu, dIu/du)
    #              = \partial F/\partial u + Action(dFdIu, Iu(uhat, v*))
    dFdu_partial = uhat * Iu * v * dx
    dFdIu = Ihat * u * v * dx
    assert dFdu == dFdu_partial + Action(dFdIu, dIu)
