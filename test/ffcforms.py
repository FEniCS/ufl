"Unit tests including all demo forms from FFC 0.4.4"

__author__ = "Anders Logg (logg@simula.no) et al."
__date__ = "2008-04-09 -- 2008-04-09"
__copyright__ = "Copyright (C) 2008 Anders Logg et al."
__license__  = "GNU GPL version 3 or any later version"

# Examples copied from the FFC demo directory, examples contributed
# by Johan Jansson, Kristian Oelgaard, Marie Rognes, and Garth Wells.

import unittest
from ufl import *

class FFC(unittest.TestCase):

    def testForm(self):

        element = FiniteElement("Lagrange", "triangle", 1)

        v = TestFunction(element)
        u = TrialFunction(element)
        f = Function(element)
        
        c = Constant("triangle")
        d = VectorConstant("triangle")
        
        a = c*dot(grad(v), grad(u))*dx
        L = dot(d, grad(v))*dx

    def testElasticity(self):

        element = VectorElement("Lagrange", "tetrahedron", 1)

        v = TestFunction(element)
        u = TrialFunction(element)

        def eps(v):
            return grad(v) + transp(grad(v))

        a = 0.25*dot(eps(v), eps(u))*dx

    def testEnergyNorm(self):

        element = FiniteElement("Lagrange", "tetrahedron", 1)

        v = Function(element)
        a = (v*v + dot(grad(v), grad(v)))*dx

    def testEquation(self):

        element = FiniteElement("Lagrange", "triangle", 1)

        k = 0.1

        v = TestFunction(element)
        u = TrialFunction(element)
        u0 = Function(element)

        F = v*(u - u0)*dx + k*dot(grad(v), grad(0.5*(u0 + u)))*dx

        a = lhs(F)
        L = rhs(F)

    def testFunctionOperators(self):

        element = FiniteElement("Lagrange", "triangle", 1)

        v = TestFunction(element)
        u = TrialFunction(element)
        f = Function(element)
        g = Function(element)

        a = sqrt(1/modulus(1/f))*sqrt(g)*dot(grad(v), grad(u))*dx + v*u*sqrt(f*g)*g*dx

    def testHeat(self):

        element = FiniteElement("Lagrange", "triangle", 1)

        v  = TestFunction(element)  # Test function
        u1 = TrialFunction(element) # Value at t_n
        u0 = Function(element)      # Value at t_n-1
        c  = Function(element)      # Heat conductivity
        f  = Function(element)      # Heat source
        k  = Constant("triangle")   # Time step

        a = v*u1*dx + k*c*dot(grad(v), grad(u1))*dx
        L = v*u0*dx + k*v*f*dx

    def testMass(self):

        element = FiniteElement("Lagrange", "tetrahedron", 3)
        
        v = TestFunction(element)
        u = TrialFunction(element)
    
        a = v*u*dx

    def testMixedMixedElement(self):

        P3 = FiniteElement("Lagrange", "triangle", 3)

        element = (P3 + P3) + (P3 + P3)

    def testMixedPoisson(self):

        q = 1

        BDM = FiniteElement("Brezzi-Douglas-Marini", "triangle", q)
        DG  = FiniteElement("Discontinuous Lagrange", "triangle", q - 1)

        mixed_element = BDM + DG

        (tau, w) = TestFunctions(mixed_element)
        (sigma, u) = TrialFunctions(mixed_element)

        f = Function(DG)

        a = (dot(tau, sigma) - div(tau)*u + w*div(sigma))*dx
        L = w*f*dx

    def testNavierStokes(self):

        element = VectorElement("Lagrange", "tetrahedron", 1)
        
        v = TestFunction(element)
        u = TrialFunction(element)

        w = Function(element)

        a = v[i]*w[j]*Dx(u[i], j)*dx

    def testNeumannProblem(self):

        element = VectorElement("Lagrange", "triangle", 1)

        v = TestFunction(element)
        u = TrialFunction(element)
        f = Function(element)
        g = Function(element)

        a = dot(grad(v), grad(u))*dx
        L = dot(v, f)*dx + dot(v, g)*ds

    def testOptimization(self):

        element = FiniteElement("Lagrange", "triangle", 3)

        v = TestFunction(element)
        u = TrialFunction(element)
        f = Function(element)

        a = dot(grad(v), grad(u))*dx
        L = v*f*dx

    def testP5tet(self):

        element = FiniteElement("Lagrange", "tetrahedron", 5)

    def testP5tri(self):

        element = FiniteElement("Lagrange", "triangle", 5)

    def testPoissonDG(self):

        # Elements
        element = FiniteElement("Discontinuous Lagrange", "triangle", 1)

        # Test and trial functions
        v = TestFunction(element)
        u = TrialFunction(element)
        f = Function(element)

        # Normal component, mesh size and right-hand side
        n = FacetNormal("triangle")
        h = MeshSize("triangle")

        # Neumann boundary conditions
        gN = Function(element)

        # Parameters
        alpha = 4.0
        gamma = 8.0

        # Bilinear form
        a = dot(grad(v), grad(u))*dx \
          - dot(avg(grad(v)), jump(u, n))*dS \
          - dot(jump(v, n), avg(grad(u)))*dS \
          + alpha/h('+')*dot(jump(v, n), jump(u, n))*dS \
          - dot(grad(v), mult(u,n))*ds \
          - dot(mult(v,n), grad(u))*ds \
          + gamma/h*v*u*ds

        # Linear form
        L = v*f*dx + v*gN*ds

    def testPoisson(self):

        element = FiniteElement("Lagrange", "triangle", 1)

        v = TestFunction(element)
        u = TrialFunction(element)
        f = Function(element)

        a = dot(grad(v), grad(u))*dx
        L = v*f*dx

    def testPoissonSystem(self):

        element = VectorElement("Lagrange", "triangle", 1)

        v = TestFunction(element)
        u = TrialFunction(element)
        f = Function(element)

        a = dot(grad(v), grad(u))*dx
        L = dot(v, f)*dx

    def testProjection(self):

        P0 = FiniteElement("Discontinuous Lagrange", "triangle", 0)
        P1 = FiniteElement("Lagrange", "triangle", 1)
        P2 = FiniteElement("Lagrange", "triangle", 2)

        v = TestFunction(P1)
        f = Function(P1)

        pi0 = Projection(P0)
        pi1 = Projection(P1)
        pi2 = Projection(P2)

        a = v*(pi0(f) + pi1(f) + pi2(f))*dx

    def testQuadratureElement(self):

        element = FiniteElement("Lagrange", "triangle", 2)
        QE = QuadratureElement("triangle", 3)
        sig = VectorQuadratureElement("triangle", 3)

        v = TestFunction(element)
        u = TrialFunction(element)
        u0= Function(element)
        C = Function(QE)
        sig0 = Function(sig)
        f = Function(element)

        a = v.dx(i)*C*u.dx(i)*dx + v.dx(i)*2*u0*u*u0.dx(i)*dx
        L = v*f*dx - dot(grad(v), sig0)*dx

    def testStokes(self):

        P2 = VectorElement("Lagrange", "triangle", 2)
        P1 = FiniteElement("Lagrange", "triangle", 1)
        TH = P2 + P1

        (v, q) = TestFunctions(TH)
        (u, p) = TrialFunctions(TH)
        
        f = Function(P2)

        a = (dot(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
        L = dot(v, f)*dx

    def testSubDomains(self):

        element = FiniteElement("CG", "tetrahedron", 1)

        v = TestFunction(element)
        u = TrialFunction(element)

        dx0 = Integral("cell", 0)
        dx1 = Integral("cell", 1)

        ds0 = Integral("exterior facet", 0)
        ds1 = Integral("exterior facet", 1)

        dS0 = Integral("interior facet", 0)
        dS1 = Integral("interior facet", 1)

        a = v*u*dx0 + 10.0*v*u*dx1 + v*u*ds0 + 2.0*v*u*ds1 + v('+')*u('+')*dS0 + 4.3*v('+')*u('+')*dS1

    def testTensorWeightedPoisson(self):

        P1 = FiniteElement("Lagrange", "triangle", 1)
        P0 = FiniteElement("Discontinuous Lagrange", "triangle", 0)

        v = TestFunction(P1)
        u = TrialFunction(P1)
        f = Function(P1)

        c00 = Function(P0)
        c01 = Function(P0)
        c10 = Function(P0)
        c11 = Function(P0)

        C = [[c00, c01], [c10, c11]]

        a = dot(grad(v), mult(C, grad(u)))*dx

    def testVectorLaplaceGradCurl(self):

        def HodgeLaplaceGradCurl(element, felement):
            (tau, v) = TestFunctions(element)
            (sigma, u) = TrialFunctions(element)
            f = Function(felement)
            a = (dot(tau, sigma) - dot(grad(tau), u) + 
                 dot(v, grad(sigma)) + dot(curl(v), curl(u)))*dx
            L = dot(v, f)*dx
            return [a, L]

        shape = "tetrahedron"
        order = 1

        GRAD = FiniteElement("Lagrange", shape, order)
        CURL = FiniteElement("Nedelec 1st kind H(curl)", shape, order-1)
        VectorLagrange = VectorElement("Lagrange", shape, order+1)

        [a, L] = HodgeLaplaceGradCurl(GRAD + CURL, VectorLagrange)

if __name__ == "__main__":
    unittest.main()
