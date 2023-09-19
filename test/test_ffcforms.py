"""Test FFC forms.

Unit tests including all demo forms from FFC 0.5.0. The forms are
modified (with comments) to work with the UFL notation which differs
from the FFC notation in some places.
"""

__author__ = "Anders Logg (logg@simula.no) et al."
__date__ = "2008-04-09 -- 2008-09-26"
__copyright__ = "Copyright (C) 2008 Anders Logg et al."
__license__ = "GNU GPL version 3 or any later version"

# Examples copied from the FFC demo directory, examples contributed
# by Johan Jansson, Kristian Oelgaard, Marie Rognes, and Garth Wells.

from ufl import (Coefficient, Constant, Dx, FacetNormal, TestFunction, TestFunctions, TrialFunction, TrialFunctions,
                 VectorConstant, avg, curl, div, dot, ds, dS, dx, grad, i, inner, j, jump, lhs, rhs, sqrt, tetrahedron,
                 triangle)
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.sobolevspace import H1, L2, HCurl, HDiv


def testConstant():
    element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)

    c = Constant("triangle")
    d = VectorConstant("triangle")

    a = c * dot(grad(v), grad(u)) * dx  # noqa: F841

    # FFC notation: L = dot(d, grad(v))*dx
    L = inner(d, grad(v)) * dx  # noqa: F841


def testElasticity():
    element = FiniteElement("Lagrange", tetrahedron, 1, (3, ), (3, ), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)

    def eps(v):
        # FFC notation: return grad(v) + transp(grad(v))
        return grad(v) + (grad(v)).T

    # FFC notation: a = 0.25*dot(eps(v), eps(u))*dx
    a = 0.25 * inner(eps(v), eps(u)) * dx  # noqa: F841


def testEnergyNorm():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), (), "identity", H1)

    v = Coefficient(element)
    a = (v * v + dot(grad(v), grad(v))) * dx  # noqa: F841


def testEquation():
    element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

    k = 0.1

    v = TestFunction(element)
    u = TrialFunction(element)
    u0 = Coefficient(element)

    F = v * (u - u0) * dx + k * dot(grad(v), grad(0.5 * (u0 + u))) * dx

    a = lhs(F)  # noqa: F841
    L = rhs(F)  # noqa: F841


def testFunctionOperators():
    element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)

    # FFC notation: a = sqrt(1/modulus(1/f))*sqrt(g)*dot(grad(v), grad(u))*dx
    # + v*u*sqrt(f*g)*g*dx
    a = sqrt(1 / abs(1 / f)) * sqrt(g) * dot(grad(v), grad(u)) * dx + v * u * sqrt(f * g) * g * dx  # noqa: F841


def testHeat():
    element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

    v = TestFunction(element)
    u1 = TrialFunction(element)
    u0 = Coefficient(element)
    c = Coefficient(element)
    f = Coefficient(element)
    k = Constant("triangle")

    a = v * u1 * dx + k * c * dot(grad(v), grad(u1)) * dx  # noqa: F841
    L = v * u0 * dx + k * v * f * dx  # noqa: F841


def testMass():
    element = FiniteElement("Lagrange", tetrahedron, 3, (), (), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)

    a = v * u * dx  # noqa: F841


def testMixedMixedElement():
    P3 = FiniteElement("Lagrange", triangle, 3, (), (), "identity", H1)
    MixedElement([[P3, P3], [P3, P3]])


def testMixedPoisson():
    q = 1

    BDM = FiniteElement("Brezzi-Douglas-Marini", triangle, q, (2, ), (2, ), "contravariant Piola", HDiv)
    DG = FiniteElement("Discontinuous Lagrange", triangle, q - 1, (), (), "identity", L2)

    mixed_element = MixedElement([BDM, DG])

    (tau, w) = TestFunctions(mixed_element)
    (sigma, u) = TrialFunctions(mixed_element)

    f = Coefficient(DG)

    a = (dot(tau, sigma) - div(tau) * u + w * div(sigma)) * dx  # noqa: F841
    L = w * f * dx  # noqa: F841


def testNavierStokes():
    element = FiniteElement("Lagrange", tetrahedron, 1, (3, ), (3, ), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)

    w = Coefficient(element)

    # FFC notation: a = v[i]*w[j]*D(u[i], j)*dx
    a = v[i] * w[j] * Dx(u[i], j) * dx  # noqa: F841


def testNeumannProblem():
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)

    # FFC notation: a = dot(grad(v), grad(u))*dx
    a = inner(grad(v), grad(u)) * dx  # noqa: F841

    # FFC notation: L = dot(v, f)*dx + dot(v, g)*ds
    L = inner(v, f) * dx + inner(v, g) * ds  # noqa: F841


def testOptimization():
    element = FiniteElement("Lagrange", triangle, 3, (), (), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    a = dot(grad(v), grad(u)) * dx  # noqa: F841
    L = v * f * dx  # noqa: F841


def testP5tet():
    FiniteElement("Lagrange", tetrahedron, 5, (), (), "identity", H1)


def testP5tri():
    FiniteElement("Lagrange", triangle, 5, (), (), "identity", H1)


def testPoissonDG():
    element = FiniteElement("Discontinuous Lagrange", triangle, 1, (), (), "identity", L2)

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

    a = inner(grad(v), grad(u)) * dx
    a -= inner(avg(grad(v)), jump(u, n)) * dS
    a -= inner(jump(v, n), avg(grad(u))) * dS
    a += alpha / h('+') * dot(jump(v, n), jump(u, n)) * dS
    a -= inner(grad(v), u * n) * ds
    a -= inner(u * n, grad(u)) * ds
    a += gamma / h * v * u * ds

    L = v * f * dx + v * gN * ds  # noqa: F841


def testPoisson():
    element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    # Note: inner() also works
    a = dot(grad(v), grad(u)) * dx  # noqa: F841
    L = v * f * dx  # noqa: F841


def testPoissonSystem():
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)

    # FFC notation: a = dot(grad(v), grad(u))*dx
    a = inner(grad(v), grad(u)) * dx  # noqa: F841

    # FFC notation: L = dot(v, f)*dx
    L = inner(v, f) * dx  # noqa: F841


def testProjection():
    # Projections are not supported by UFL and have been broken
    # in FFC for a while. For DOLFIN, the current (global) L^2
    # projection can be extended to handle also local projections.

    P1 = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

    v = TestFunction(P1)  # noqa: F841
    f = Coefficient(P1)  # noqa: F841

    # pi0 = Projection(P0)
    # pi1 = Projection(P1)
    # pi2 = Projection(P2)
    #
    # a = v*(pi0(f) + pi1(f) + pi2(f))*dx


def testQuadratureElement():
    element = FiniteElement("Lagrange", triangle, 2, (), (), "identity", H1)

    # FFC notation:
    # QE = QuadratureElement("triangle", 3)
    # sig = VectorQuadratureElement("triangle", 3)

    QE = FiniteElement("Quadrature", triangle, 3, (), (), "identity", L2)
    sig = FiniteElement("Quadrature", triangle, 3, (2, ), (2, ), "identity", L2)

    v = TestFunction(element)
    u = TrialFunction(element)
    u0 = Coefficient(element)
    C = Coefficient(QE)
    sig0 = Coefficient(sig)
    f = Coefficient(element)

    a = v.dx(i) * C * u.dx(i) * dx + v.dx(i) * 2 * u0 * u * u0.dx(i) * dx  # noqa: F841
    L = v * f * dx - dot(grad(v), sig0) * dx  # noqa: F841


def testStokes():
    # UFLException: Shape mismatch in sum.

    P2 = FiniteElement("Lagrange", triangle, 2, (2, ), (2, ), "identity", H1)
    P1 = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    TH = MixedElement([P2, P1])

    (v, q) = TestFunctions(TH)
    (u, p) = TrialFunctions(TH)

    f = Coefficient(P2)

    # FFC notation:
    # a = (dot(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
    a = (inner(grad(v), grad(u)) - div(v) * p + q * div(u)) * dx  # noqa: F841

    L = dot(v, f) * dx  # noqa: F841


def testSubDomain():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), (), "identity", H1)

    f = Coefficient(element)

    M = f * dx(2) + f * ds(5)  # noqa: F841


def testSubDomains():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), (), "identity", H1)

    v = TestFunction(element)
    u = TrialFunction(element)

    a = v * u * dx(0) + 10.0 * v * u * dx(1) + v * u * ds(0) + 2.0 * v * u * ds(1)
    a += v('+') * u('+') * dS(0) + 4.3 * v('+') * u('+') * dS(1)


def testTensorWeightedPoisson():
    # FFC notation:
    # P1 = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    # P0 = FiniteElement("Discontinuous Lagrange", triangle, 0, (), (), "identity", L2)
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

    P1 = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    P0 = FiniteElement("Discontinuous Lagrange", triangle, 0, (2, 2), (2, 2), "identity", L2)

    v = TestFunction(P1)
    u = TrialFunction(P1)
    C = Coefficient(P0)

    a = inner(grad(v), C * grad(u)) * dx  # noqa: F841


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

    shape = tetrahedron
    order = 1

    GRAD = FiniteElement("Lagrange", shape, order, (), (), "identity", H1)

    # FFC notation: CURL = FiniteElement("Nedelec", shape, order-1)
    CURL = FiniteElement("N1curl", shape, order, (3, ), (3, ), "covariant Piola", HCurl)

    VectorLagrange = FiniteElement("Lagrange", shape, order + 1, (3, ), (3, ), "identity", H1)

    [a, L] = HodgeLaplaceGradCurl(MixedElement([GRAD, CURL]), VectorLagrange)
