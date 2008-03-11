#!/usr/bin/env python

import unittest

from ufl import *
from ufl.utilities import * 


# TODO: these tests only verify that the syntax is possible, how much more can we test without a form compiler?

# TODO: add more forms, covering all UFL operators


class FormsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_source1(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element, "f")
        try:
            a = f*v*dx
            self.fail()
        except:
            pass
    
    def test_source2(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element, "f")
        try:
            a = dot(f[0],v)*dx
            self.fail()
        except:
            pass
    
    def test_source3(self):
        element = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element, "f")
        try:
            a = inner(f,v[0])*dx
            self.fail()
        except:
            pass
    
    
    def test_mass1(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        try:
            a = u[i]*v*dx
            self.fail()
        except:
            pass
    
    def test_mass2(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        try:
            a = u[i][j]
            self.fail()
        except:
            pass
    
    def test_mass3(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        try:
            a = dot(u[i],v[j])*dx
            self.fail()
        except:
            pass
    
    def test_mass4(self):
        element = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(u,v)*dx


    def test_stiffness1(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = dot(grad(u), grad(v)) * dx

    def test_stiffness2(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(grad(u), grad(v)) * dx

    def test_stiffness3(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(grad(u), grad(v)) * dx


    def test_stiffness_with_conductivity(self):
        velement = VectorElement("Lagrange", "triangle", 1)
        telement = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(velement)
        u = TrialFunction(velement)
        M = Function(telement, "M")
        a = inner(M*grad(u), grad(v)) * dx


    def test_navier_stokes(self):
        polygon = "triangle"
        velement = VectorElement("Lagrange", polygon, 2)
        pelement = FiniteElement("Lagrange", polygon, 1)
        TH = velement + pelement

        v, q = TestFunctions(TH)
        u, p = TrialFunctions(TH)

        f = Function(velement, "f")
        w = Function(velement, "w")
        Re = Constant(polygon, "w")
        dt = Constant(polygon, "dt")

        a = dot(u, v) + dt*dot(dot(w, grad(u)), v) - dt*Re*inner(grad(u), grad(v)) + dt*dot(grad(p), v)
        L = dot(f, v)
        b = dot(u, grad(q))


suite1 = unittest.makeSuite(FormsTestCase)

allsuites = unittest.TestSuite((suite1, ))

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=0).run(allsuites)

