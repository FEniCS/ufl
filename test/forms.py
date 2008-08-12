#!/usr/bin/env python

import unittest

from ufl import *
from ufl.utilities import * 

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


# TODO: these tests only verify that the syntax is possible, how much more can we test without a form compiler?

# TODO: add more forms, covering all UFL operators


class FormsTestCase(unittest.TestCase):

    def setUp(self):
        pass
     
    def test_separated_dx(self):
        "Tests automatic summation of integrands over same domain."
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element, "f")
        a = f*v*dx + 2*v*ds + 3*v*dx + 7*v*ds + 3*v*dx2 + 7*v*dx2
        b = (f*v + 3*v)*dx + (2*v + 7*v)*ds + (3*v + 7*v)*dx2 
        self.assertTrue(repr(a) == repr(b))

    def test_source1(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element, "f")
        a = f*v*dx
        
    def test_source2(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element, "f")
        a = dot(f,v)*dx
        
    def test_source3(self):
        element = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element, "f")
        a = inner(f,v)*dx


    def test_mass1(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = u*v*dx

    def test_mass2(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = u*v*dx
        
    def test_mass3(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = dot(u,v)*dx
        
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
