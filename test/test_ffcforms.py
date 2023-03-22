#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
"""Unit tests including all demo forms from FFC 0.5.0. The forms are
modified (with comments) to work with the UFL notation which differs
from the FFC notation in some places."""

__author__ = "Anders Logg (logg@simula.no) et al."
__date__ = "2008-04-09 -- 2008-09-26"
__copyright__ = "Copyright (C) 2008 Anders Logg et al."
__license__ = "GNU GPL version 3 or any later version"

# Examples copied from the FFC demo directory, examples contributed
# by Johan Jansson, Kristian Oelgaard, Marie Rognes, and Garth Wells.

import pytest
from ufl_legacy import *


def testConstant():

    element = FiniteElement("Lagrange", "triangle", 1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    c = Constant("triangle")
    d = VectorConstant("triangle")

    a = c * dot(grad(v), grad(u)) * dx

    # FFC notation: L = dot(d, grad(v))*dx
    L = inner(d, grad(v)) * dx


def testElasticity():

    element = VectorElement("Lagrange", "tetrahedron", 1)

    v = TestFunction(element)
    u = TrialFunction(element)

    def eps(v):
        # FFC notation: return grad(v) + transp(grad(v))
        return grad(v) + (grad(v)).T

    # FFC notation: a = 0.25*dot(eps(v), eps(u))*dx
    a = 0.25 * inner(eps(v), eps(u)) * dx


def testEnergyNorm():

    element = FiniteElement("Lagrange", "tetrahedron", 1)

    v = Coefficient(element)
    a = (v * v + dot(grad(v), grad(v))) * dx


def testEquation():

    element = FiniteElement("Lagrange", "triangle", 1)

    k = 0.1

    v = TestFunction(element)
    u = TrialFunction(element)
    u0 = Coefficient(element)

    F = v * (u - u0) * dx + k * dot(grad(v), grad(0.5 * (u0 + u))) * dx

    a = lhs(F)
    L = rhs(F)


def testFunctionOperators():

    element = FiniteElement("Lagrange", "triangle", 1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)

    # FFC notation: a = sqrt(1/modulus(1/f))*sqrt(g)*dot(grad(v), grad(u))*dx
    # + v*u*sqrt(f*g)*g*dx
    a = sqrt(1 / abs(1 / f)) * sqrt(g) * \
        dot(grad(v), grad(u)) * dx + v * u * sqrt(f * g) * g * dx


def testHeat():

    element = FiniteElement("Lagrange", "triangle", 1)

    v = TestFunction(element)
    u1 = TrialFunction(element)
    u0 = Coefficient(element)
    c = Coefficient(element)
    f = Coefficient(element)
    k = Constant("triangle")

    a = v * u1 * dx + k * c * dot(grad(v), grad(u1)) * dx
    L = v * u0 * dx + k * v * f * dx


def testMass():

    element = FiniteElement("Lagrange", "tetrahedron", 3)

    v = TestFunction(element)
    u = TrialFunction(element)

    a = v * u * dx


def testMixedMixedElement():

    P3 = FiniteElement("Lagrange", "triangle", 3)

    element = (P3 * P3) * (P3 * P3)


def testMixedPoisson():

    q = 1

    BDM = FiniteElement("Brezzi-Douglas-Marini", "triangle", q)
    DG = FiniteElement("Discontinuous Lagrange", "triangle", q - 1)

    mixed_element = BDM * DG

    (tau, w) = TestFunctions(mixed_element)
    (sigma, u) = TrialFunctions(mixed_element)

    f = Coefficient(DG)

    a = (dot(tau, sigma) - div(tau) * u + w * div(sigma)) * dx
    L = w * f * dx


def testNavierStokes():

    element = VectorElement("Lagrange", "tetrahedron", 1)

    v = TestFunction(element)
    u = TrialFunction(element)

    w = Coefficient(element)

    # FFC notation: a = v[i]*w[j]*D(u[i], j)*dx
    a = v[i] * w[j] * Dx(u[i], j) * dx


def testNeumannProblem():

    element = VectorElement("Lagrange", "triangle", 1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)

    # FFC notation: a = dot(grad(v), grad(u))*dx
    a = inner(grad(v), grad(u)) * dx

    # FFC notation: L = dot(v, f)*dx + dot(v, g)*ds
    L = inner(v, f) * dx + inner(v, g) * ds


def testOptimization():

    element = FiniteElement("Lagrange", "triangle", 3)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    a = dot(grad(v), grad(u)) * dx
    L = v * f * dx


def testP5tet():

    element = FiniteElement("Lagrange", tetrahedron, 5)


def testP5tri():

    element = FiniteElement("Lagrange", triangle, 5)


def testPoissonDG():

    element = FiniteElement("Discontinuous Lagrange", triangle, 1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    n = FacetNormal(triangle)

    # FFC notation: h = MeshSize("triangle"), not supported by UFL
    h = Constant(triangle)

    gN = Coefficient(element)

    alpha = 4.0
    gamma = 8.0

    # FFC notation
    # a = dot(grad(v), grad(u))*dx \
    #    - dot(avg(grad(v)), jump(u, n))*dS \
    #    - dot(jump(v, n), avg(grad(u)))*dS \
    #    + alpha/h('+')*dot(jump(v, n), jump(u, n))*dS \
    #    - dot(grad(v), mult(u,n))*ds \
    #    - dot(mult(v,n), grad(u))*ds \
    #    + gamma/h*v*u*ds

    a = inner(grad(v), grad(u)) * dx \
        - inner(avg(grad(v)), jump(u, n)) * dS \
        - inner(jump(v, n), avg(grad(u))) * dS \
        + alpha / h('+') * dot(jump(v, n), jump(u, n)) * dS \
        - inner(grad(v), u * n) * ds \
        - inner(u * n, grad(u)) * ds \
        + gamma / h * v * u * ds

    L = v * f * dx + v * gN * ds


def testPoisson():

    element = FiniteElement("Lagrange", "triangle", 1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    # Note: inner() also works
    a = dot(grad(v), grad(u)) * dx
    L = v * f * dx


def testPoissonSystem():

    element = VectorElement("Lagrange", "triangle", 1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    # FFC notation: a = dot(grad(v), grad(u))*dx
    a = inner(grad(v), grad(u)) * dx

    # FFC notation: L = dot(v, f)*dx
    L = inner(v, f) * dx


def testProjection():

    # Projections are not supported by UFL and have been broken
    # in FFC for a while. For DOLFIN, the current (global) L^2
    # projection can be extended to handle also local projections.

    P0 = FiniteElement("Discontinuous Lagrange", "triangle", 0)
    P1 = FiniteElement("Lagrange", "triangle", 1)
    P2 = FiniteElement("Lagrange", "triangle", 2)

    v = TestFunction(P1)
    f = Coefficient(P1)

    # pi0 = Projection(P0)
    # pi1 = Projection(P1)
    # pi2 = Projection(P2)
    #
    # a = v*(pi0(f) + pi1(f) + pi2(f))*dx


def testQuadratureElement():

    element = FiniteElement("Lagrange", "triangle", 2)

    # FFC notation:
    # QE = QuadratureElement("triangle", 3)
    # sig = VectorQuadratureElement("triangle", 3)

    QE = FiniteElement("Quadrature", "triangle", 3)
    sig = VectorElement("Quadrature", "triangle", 3)

    v = TestFunction(element)
    u = TrialFunction(element)
    u0 = Coefficient(element)
    C = Coefficient(QE)
    sig0 = Coefficient(sig)
    f = Coefficient(element)

    a = v.dx(i) * C * u.dx(i) * dx + v.dx(i) * 2 * u0 * u * u0.dx(i) * dx
    L = v * f * dx - dot(grad(v), sig0) * dx


def testStokes():

    # UFLException: Shape mismatch in sum.

    P2 = VectorElement("Lagrange", "triangle", 2)
    P1 = FiniteElement("Lagrange", "triangle", 1)
    TH = P2 * P1

    (v, q) = TestFunctions(TH)
    (u, p) = TrialFunctions(TH)

    f = Coefficient(P2)

    # FFC notation:
    # a = (dot(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
    a = (inner(grad(v), grad(u)) - div(v) * p + q * div(u)) * dx

    L = dot(v, f) * dx


def testSubDomain():

    element = FiniteElement("CG", "tetrahedron", 1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    M = f * dx(2) + f * ds(5)


def testSubDomains():

    element = FiniteElement("CG", "tetrahedron", 1)

    v = TestFunction(element)
    u = TrialFunction(element)

    a = v * u * dx(0) + 10.0 * v * u * dx(1) + v * u * ds(0) + 2.0 * v * u * \
        ds(1) + v('+') * u('+') * dS(0) + 4.3 * v('+') * u('+') * dS(1)


def testTensorWeightedPoisson():

    # FFC notation:
    # P1 = FiniteElement("Lagrange", "triangle", 1)
    # P0 = FiniteElement("Discontinuous Lagrange", "triangle", 0)
    #
    # v = TestFunction(P1)
    # u = TrialFunction(P1)
    # f = Coefficient(P1)
    #
    # c00 = Coefficient(P0)
    # c01 = Coefficient(P0)
    # c10 = Coefficient(P0)
    # c11 = Coefficient(P0)
    #
    # C = [[c00, c01], [c10, c11]]
    #
    # a = dot(grad(v), mult(C, grad(u)))*dx

    P1 = FiniteElement("Lagrange", "triangle", 1)
    P0 = TensorElement("Discontinuous Lagrange", "triangle", 0, shape=(2, 2))

    v = TestFunction(P1)
    u = TrialFunction(P1)
    C = Coefficient(P0)

    a = inner(grad(v), C * grad(u)) * dx


def testVectorLaplaceGradCurl():

    def HodgeLaplaceGradCurl(element, felement):
        (tau, v) = TestFunctions(element)
        (sigma, u) = TrialFunctions(element)
        f = Coefficient(felement)

        # FFC notation: a = (dot(tau, sigma) - dot(grad(tau), u) + dot(v,
        # grad(sigma)) + dot(curl(v), curl(u)))*dx
        a = (inner(tau, sigma) - inner(grad(tau), u) +
             inner(v, grad(sigma)) + inner(curl(v), curl(u))) * dx

        # FFC notation: L = dot(v, f)*dx
        L = inner(v, f) * dx

        return [a, L]

    shape = "tetrahedron"
    order = 1

    GRAD = FiniteElement("Lagrange", shape, order)

    # FFC notation: CURL = FiniteElement("Nedelec", shape, order-1)
    CURL = FiniteElement("N1curl", shape, order)

    VectorLagrange = VectorElement("Lagrange", shape, order + 1)

    [a, L] = HodgeLaplaceGradCurl(GRAD * CURL, VectorLagrange)
