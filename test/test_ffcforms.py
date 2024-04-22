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

from ufl import (
    Coefficient,
    Constant,
    Dx,
    FacetNormal,
    FunctionSpace,
    Mesh,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    VectorConstant,
    avg,
    curl,
    div,
    dot,
    dS,
    ds,
    dx,
    grad,
    i,
    inner,
    j,
    jump,
    lhs,
    rhs,
    sqrt,
    tetrahedron,
    triangle,
)
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pullback import contravariant_piola, covariant_piola, identity_pullback
from ufl.sobolevspace import H1, L2, HCurl, HDiv


def testConstant():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)

    c = Constant(domain)
    d = VectorConstant(domain)

    _ = c * dot(grad(v), grad(u)) * dx

    # FFC notation: L = dot(d, grad(v))*dx
    _ = inner(d, grad(v)) * dx


def testElasticity():
    element = FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)

    def eps(v):
        # FFC notation: return grad(v) + transp(grad(v))
        return grad(v) + (grad(v)).T

    # FFC notation: a = 0.25*dot(eps(v), eps(u))*dx
    _ = 0.25 * inner(eps(v), eps(u)) * dx


def testEnergyNorm():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = Coefficient(space)
    _ = (v * v + dot(grad(v), grad(v))) * dx


def testEquation():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    k = 0.1

    v = TestFunction(space)
    u = TrialFunction(space)
    u0 = Coefficient(space)

    F = v * (u - u0) * dx + k * dot(grad(v), grad(0.5 * (u0 + u))) * dx

    _ = lhs(F)
    _ = rhs(F)


def testFunctionOperators():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)

    # FFC notation: a = sqrt(1/modulus(1/f))*sqrt(g)*dot(grad(v), grad(u))*dx
    # + v*u*sqrt(f*g)*g*dx
    _ = sqrt(1 / abs(1 / f)) * sqrt(g) * dot(grad(v), grad(u)) * dx + v * u * sqrt(f * g) * g * dx


def testHeat():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u1 = TrialFunction(space)
    u0 = Coefficient(space)
    c = Coefficient(space)
    f = Coefficient(space)
    k = Constant(domain)

    _ = v * u1 * dx + k * c * dot(grad(v), grad(u1)) * dx
    _ = v * u0 * dx + k * v * f * dx


def testMass():
    element = FiniteElement("Lagrange", tetrahedron, 3, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    _ = v * u * dx


def testMixedMixedElement():
    P3 = FiniteElement("Lagrange", triangle, 3, (), identity_pullback, H1)
    MixedElement([[P3, P3], [P3, P3]])


def testMixedPoisson():
    q = 1
    BDM = FiniteElement("Brezzi-Douglas-Marini", triangle, q, (2,), contravariant_piola, HDiv)
    DG = FiniteElement("Discontinuous Lagrange", triangle, q - 1, (), identity_pullback, L2)

    mixed_element = MixedElement([BDM, DG])
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, mixed_element)

    (tau, w) = TestFunctions(space)
    (sigma, u) = TrialFunctions(space)
    f = Coefficient(FunctionSpace(domain, DG))
    _ = (dot(tau, sigma) - div(tau) * u + w * div(sigma)) * dx
    _ = w * f * dx


def testNavierStokes():
    element = FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    w = Coefficient(space)

    # FFC notation: a = v[i]*w[j]*D(u[i], j)*dx
    _ = v[i] * w[j] * Dx(u[i], j) * dx


def testNeumannProblem():
    element = FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)

    # FFC notation: a = dot(grad(v), grad(u))*dx
    _ = inner(grad(v), grad(u)) * dx

    # FFC notation: L = dot(v, f)*dx + dot(v, g)*ds
    _ = inner(v, f) * dx + inner(v, g) * ds


def testOptimization():
    element = FiniteElement("Lagrange", triangle, 3, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)
    _ = dot(grad(v), grad(u)) * dx
    _ = v * f * dx


def testP5tet():
    FiniteElement("Lagrange", tetrahedron, 5, (), identity_pullback, H1)


def testP5tri():
    FiniteElement("Lagrange", triangle, 5, (), identity_pullback, H1)


def testPoissonDG():
    element = FiniteElement("Discontinuous Lagrange", triangle, 1, (), identity_pullback, L2)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)
    n = FacetNormal(domain)

    # FFC notation: h = MeshSize(domain), not supported by UFL
    h = Constant(domain)

    gN = Coefficient(space)
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
    a += alpha / h("+") * dot(jump(v, n), jump(u, n)) * dS
    a -= inner(grad(v), u * n) * ds
    a -= inner(u * n, grad(u)) * ds
    a += gamma / h * v * u * ds

    _ = v * f * dx + v * gN * ds


def testPoisson():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)

    # Note: inner() also works
    _ = dot(grad(v), grad(u)) * dx
    _ = v * f * dx


def testPoissonSystem():
    element = FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)

    # FFC notation: a = dot(grad(v), grad(u))*dx
    _ = inner(grad(v), grad(u)) * dx

    # FFC notation: L = dot(v, f)*dx
    _ = inner(v, f) * dx


def testProjection():
    # Projections are not supported by UFL and have been broken
    # in FFC for a while. For DOLFIN, the current (global) L^2
    # projection can be extended to handle also local projections.

    P1 = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, P1)
    _ = TestFunction(space)
    _ = Coefficient(space)

    # pi0 = Projection(P0)
    # pi1 = Projection(P1)
    # pi2 = Projection(P2)
    #
    # a = v*(pi0(f) + pi1(f) + pi2(f))*dx


def testQuadratureElement():
    element = FiniteElement("Lagrange", triangle, 2, (), identity_pullback, H1)

    # FFC notation:
    # QE = QuadratureElement(triangle, 3)
    # sig = VectorQuadratureElement(triangle, 3)

    QE = FiniteElement("Quadrature", triangle, 3, (), identity_pullback, L2)
    sig = FiniteElement("Quadrature", triangle, 3, (2,), identity_pullback, L2)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    u0 = Coefficient(space)
    C = Coefficient(FunctionSpace(domain, QE))
    sig0 = Coefficient(FunctionSpace(domain, sig))
    f = Coefficient(space)

    _ = v.dx(i) * C * u.dx(i) * dx + v.dx(i) * 2 * u0 * u * u0.dx(i) * dx
    _ = v * f * dx - dot(grad(v), sig0) * dx


def testStokes():
    # UFLException: Shape mismatch in sum.

    P2 = FiniteElement("Lagrange", triangle, 2, (2,), identity_pullback, H1)
    P1 = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    TH = MixedElement([P2, P1])

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    th_space = FunctionSpace(domain, TH)
    p2_space = FunctionSpace(domain, P2)

    (v, q) = TestFunctions(th_space)
    (u, p) = TrialFunctions(th_space)
    f = Coefficient(p2_space)

    # FFC notation:
    # a = (dot(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
    _ = (inner(grad(v), grad(u)) - div(v) * p + q * div(u)) * dx
    _ = dot(v, f) * dx


def testSubDomain():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    _ = f * dx(2) + f * ds(5)


def testSubDomains():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    a = v * u * dx(0) + 10.0 * v * u * dx(1) + v * u * ds(0) + 2.0 * v * u * ds(1)
    a += v("+") * u("+") * dS(0) + 4.3 * v("+") * u("+") * dS(1)


def testTensorWeightedPoisson():
    # FFC notation:
    # P1 = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    # P0 = FiniteElement("Discontinuous Lagrange", triangle, 0, (), identity_pullback, L2)
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

    P1 = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    P0 = FiniteElement("Discontinuous Lagrange", triangle, 0, (2, 2), identity_pullback, L2)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    p1_space = FunctionSpace(domain, P1)
    p0_space = FunctionSpace(domain, P0)

    v = TestFunction(p1_space)
    u = TrialFunction(p1_space)
    C = Coefficient(p0_space)
    _ = inner(grad(v), C * grad(u)) * dx


def testVectorLaplaceGradCurl():
    def HodgeLaplaceGradCurl(space, fspace):
        (tau, v) = TestFunctions(space)
        (sigma, u) = TrialFunctions(space)
        f = Coefficient(fspace)

        # FFC notation: a = (dot(tau, sigma) - dot(grad(tau), u) + dot(v,
        # grad(sigma)) + dot(curl(v), curl(u)))*dx
        a = (
            inner(tau, sigma)
            - inner(grad(tau), u)
            + inner(v, grad(sigma))
            + inner(curl(v), curl(u))
        ) * dx

        # FFC notation: L = dot(v, f)*dx
        L = inner(v, f) * dx

        return [a, L]

    shape = tetrahedron
    order = 1

    GRAD = FiniteElement("Lagrange", shape, order, (), identity_pullback, H1)

    # FFC notation: CURL = FiniteElement("Nedelec", shape, order-1)
    CURL = FiniteElement("N1curl", shape, order, (3,), covariant_piola, HCurl)

    VectorLagrange = FiniteElement("Lagrange", shape, order + 1, (3,), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", shape, 1, (3,), identity_pullback, H1))

    [a, L] = HodgeLaplaceGradCurl(
        FunctionSpace(domain, MixedElement([GRAD, CURL])), FunctionSpace(domain, VectorLagrange)
    )
