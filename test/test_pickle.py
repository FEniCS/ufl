"""Pickle all the unit test forms from FFC 0.5.0"""

__author__ = "Anders Logg (logg@simula.no) et al."
__date__ = "2008-04-09 -- 2008-09-26"
__copyright__ = "Copyright (C) 2008 Anders Logg et al."
__license__ = "GNU GPL version 3 or any later version"

# Examples copied from the FFC demo directory, examples contributed
# by Johan Jansson, Kristian Oelgaard, Marie Rognes, and Garth Wells.

import pickle

from ufl import (Coefficient, Constant, Dx, FacetNormal, FunctionSpace, Identity, Mesh, TestFunction, TestFunctions,
                 TrialFunction, TrialFunctions, VectorConstant, avg, curl, div, dot, dS, ds, dx, grad, i, inner, j,
                 jump, lhs, rhs, sqrt, tetrahedron, triangle)
from ufl.algorithms import compute_form_data
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pull_back import contravariant_piola, covariant_piola, identity_pull_back
from ufl.sobolevspace import H1, L2, HCurl, HDiv

p = pickle.HIGHEST_PROTOCOL


def testConstant():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)

    c = Constant(domain)
    d = VectorConstant(domain)

    a = c * dot(grad(v), grad(u)) * dx

    # FFC notation: L = dot(d, grad(v))*dx
    L = inner(d, grad(v)) * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testElasticity():
    element = FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)

    def eps(v):
        # FFC notation: return grad(v) + transp(grad(v))
        return grad(v) + (grad(v)).T

    # FFC notation: a = 0.25*dot(eps(v), eps(u))*dx
    a = 0.25 * inner(eps(v), eps(u)) * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)

    assert a.signature() == a_restore.signature()


def testEnergyNorm():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = Coefficient(space)
    a = (v * v + dot(grad(v), grad(v))) * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)

    assert a.signature() == a_restore.signature()


def testEquation():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    k = 0.1

    v = TestFunction(space)
    u = TrialFunction(space)
    u0 = Coefficient(space)

    F = v * (u - u0) * dx + k * dot(grad(v), grad(0.5 * (u0 + u))) * dx

    a = lhs(F)
    L = rhs(F)

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testFunctionOperators():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)

    # FFC notation: a = sqrt(1/modulus(1/f))*sqrt(g)*dot(grad(v), grad(u))*dx
    # + v*u*sqrt(f*g)*g*dx
    a = sqrt(1 / abs(1 / f)) * sqrt(g) * \
        dot(grad(v), grad(u)) * dx + v * u * sqrt(f * g) * g * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)

    assert a.signature() == a_restore.signature()


def testHeat():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u1 = TrialFunction(space)
    u0 = Coefficient(space)
    c = Coefficient(space)
    f = Coefficient(space)
    k = Constant(domain)

    a = v * u1 * dx + k * c * dot(grad(v), grad(u1)) * dx
    L = v * u0 * dx + k * v * f * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testMass():
    element = FiniteElement("Lagrange", tetrahedron, 3, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)

    a = v * u * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)

    assert a.signature() == a_restore.signature()


def testMixedMixedElement():
    P3 = FiniteElement("Lagrange", triangle, 3, (), identity_pull_back, H1)

    element = MixedElement([[P3, P3], [P3, P3]])

    element_pickle = pickle.dumps(element, p)
    element_restore = pickle.loads(element_pickle)

    assert element == element_restore


def testMixedPoisson():
    q = 1

    BDM = FiniteElement("Brezzi-Douglas-Marini", triangle, q, (2, ), contravariant_piola, HDiv)
    DG = FiniteElement("Discontinuous Lagrange", triangle, q - 1, (), identity_pull_back, L2)

    mixed_element = MixedElement([BDM, DG])
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    mixed_space = FunctionSpace(domain, mixed_element)
    dg_space = FunctionSpace(domain, DG)

    (tau, w) = TestFunctions(mixed_space)
    (sigma, u) = TrialFunctions(mixed_space)

    f = Coefficient(dg_space)

    a = (dot(tau, sigma) - div(tau) * u + w * div(sigma)) * dx
    L = w * f * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testNavierStokes():
    element = FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)

    w = Coefficient(space)

    # FFC notation: a = v[i]*w[j]*D(u[i], j)*dx
    a = v[i] * w[j] * Dx(u[i], j) * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)

    assert a.signature() == a_restore.signature()


def testNeumannProblem():
    element = FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)

    # FFC notation: a = dot(grad(v), grad(u))*dx
    a = inner(grad(v), grad(u)) * dx

    # FFC notation: L = dot(v, f)*dx + dot(v, g)*ds
    L = inner(v, f) * dx + inner(v, g) * ds

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testOptimization():
    element = FiniteElement("Lagrange", triangle, 3, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)

    a = dot(grad(v), grad(u)) * dx
    L = v * f * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testP5tet():
    element = FiniteElement("Lagrange", tetrahedron, 5, (), identity_pull_back, H1)

    element_pickle = pickle.dumps(element, p)
    element_restore = pickle.loads(element_pickle)

    assert element == element_restore


def testP5tri():
    element = FiniteElement("Lagrange", triangle, 5, (), identity_pull_back, H1)

    element_pickle = pickle.dumps(element, p)
    pickle.loads(element_pickle)


def testPoissonDG():
    element = FiniteElement("Discontinuous Lagrange", triangle, 1, (), identity_pull_back, L2)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
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

    a = inner(grad(v), grad(u)) * dx \
        - inner(avg(grad(v)), jump(u, n)) * dS \
        - inner(jump(v, n), avg(grad(u))) * dS \
        + alpha / h('+') * dot(jump(v, n), jump(u, n)) * dS \
        - inner(grad(v), u * n) * ds \
        - inner(u * n, grad(u)) * ds \
        + gamma / h * v * u * ds

    L = v * f * dx + v * gN * ds

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testPoisson():
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)

    # Note: inner() also works
    a = dot(grad(v), grad(u)) * dx
    L = v * f * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testPoissonSystem():
    element = FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)

    # FFC notation: a = dot(grad(v), grad(u))*dx
    a = inner(grad(v), grad(u)) * dx

    # FFC notation: L = dot(v, f)*dx
    L = inner(v, f) * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testQuadratureElement():
    element = FiniteElement("Lagrange", triangle, 2, (), identity_pull_back, H1)

    # FFC notation:
    # QE = QuadratureElement(triangle, 3)
    # sig = VectorQuadratureElement(triangle, 3)

    QE = FiniteElement("Quadrature", triangle, 3, (), identity_pull_back, L2)
    sig = FiniteElement("Quadrature", triangle, 3, (2, ), identity_pull_back, L2)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)
    qe_space = FunctionSpace(domain, QE)
    sig_space = FunctionSpace(domain, sig)

    v = TestFunction(space)
    u = TrialFunction(space)
    u0 = Coefficient(space)
    C = Coefficient(qe_space)
    sig0 = Coefficient(sig_space)
    f = Coefficient(space)

    a = v.dx(i) * C * u.dx(i) * dx + v.dx(i) * 2 * u0 * u * u0.dx(i) * dx
    L = v * f * dx - dot(grad(v), sig0) * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testStokes():
    # UFLException: Shape mismatch in sum.

    P2 = FiniteElement("Lagrange", triangle, 2, (2, ), identity_pull_back, H1)
    P1 = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    TH = MixedElement([P2, P1])

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    th_space = FunctionSpace(domain, TH)
    p2_space = FunctionSpace(domain, P2)

    (v, q) = TestFunctions(th_space)
    (u, r) = TrialFunctions(th_space)

    f = Coefficient(p2_space)

    # FFC notation:
    # a = (dot(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
    a = (inner(grad(v), grad(u)) - div(v) * r + q * div(u)) * dx

    L = dot(v, f) * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testSubDomain():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    f = Coefficient(space)

    M = f * dx(2) + f * ds(5)

    M_pickle = pickle.dumps(M, p)
    M_restore = pickle.loads(M_pickle)

    assert M.signature() == M_restore.signature()


def testSubDomains():
    element = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)

    a = v * u * dx(0) + 10.0 * v * u * dx(1) + v * u * ds(0) + 2.0 * v * u * \
        ds(1) + v('+') * u('+') * dS(0) + 4.3 * v('+') * u('+') * dS(1)

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)

    assert a.signature() == a_restore.signature()


def testTensorWeightedPoisson():
    # FFC notation:
    # P1 = FiniteElement("Lagrange", triangle, 1)
    # P0 = FiniteElement("Discontinuous Lagrange", triangle, 0)
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

    P1 = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    P0 = FiniteElement("Discontinuous Lagrange", triangle, 0, (2, 2), identity_pull_back, L2)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    p1_space = FunctionSpace(domain, P1)
    p0_space = FunctionSpace(domain, P0)

    v = TestFunction(p1_space)
    u = TrialFunction(p1_space)
    C = Coefficient(p0_space)

    a = inner(grad(v), C * grad(u)) * dx

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)

    assert a.signature() == a_restore.signature()


def testVectorLaplaceGradCurl():
    def HodgeLaplaceGradCurl(space, fspace):
        (tau, v) = TestFunctions(space)
        (sigma, u) = TrialFunctions(space)
        f = Coefficient(fspace)

        # FFC notation: a = (dot(tau, sigma) - dot(grad(tau), u) + dot(v,
        # grad(sigma)) + dot(curl(v), curl(u)))*dx
        a = (inner(tau, sigma) - inner(grad(tau), u) +
             inner(v, grad(sigma)) + inner(curl(v), curl(u))) * dx

        # FFC notation: L = dot(v, f)*dx
        L = inner(v, f) * dx

        return [a, L]

    shape = tetrahedron
    order = 1

    GRAD = FiniteElement("Lagrange", shape, order, (), identity_pull_back, H1)

    # FFC notation: CURL = FiniteElement("Nedelec", shape, order-1)
    CURL = FiniteElement("N1curl", shape, order, (3, ), covariant_piola, HCurl)

    VectorLagrange = FiniteElement("Lagrange", shape, order + 1, (3, ), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", shape, 1, (3, ), identity_pull_back, H1))

    [a, L] = HodgeLaplaceGradCurl(FunctionSpace(domain, MixedElement([GRAD, CURL])),
                                  FunctionSpace(domain, VectorLagrange))

    a_pickle = pickle.dumps(a, p)
    a_restore = pickle.loads(a_pickle)
    L_pickle = pickle.dumps(L, p)
    L_restore = pickle.loads(L_pickle)

    assert a.signature() == a_restore.signature()
    assert L.signature() == L_restore.signature()


def testIdentity():
    i = Identity(2)
    i_pickle = pickle.dumps(i, p)
    i_restore = pickle.loads(i_pickle)
    assert i == i_restore


def testFormData():
    element = FiniteElement("Lagrange", tetrahedron, 3, (), identity_pull_back, H1)
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)

    a = v * u * dx

    form_data = compute_form_data(a)

    form_data_pickle = pickle.dumps(form_data, p)
    form_data_restore = pickle.loads(form_data_pickle)

    assert str(form_data) == str(form_data_restore)
