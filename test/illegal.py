#!/usr/bin/env python

import unittest

from ufl import *
from ufl.algorithms import * 

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


# TODO: these tests only verify that the syntax is possible, how much more can we test without a form compiler?

# TODO: add more forms, covering all UFL operators


class IllegalExpressionsTestCase(unittest.TestCase):
    
    def setUp(self):
        self.selement = FiniteElement("Lagrange", "triangle", 1)
        self.velement = VectorElement("Lagrange", "triangle", 1)
        self.a = BasisFunction(self.selement)
        self.b = BasisFunction(self.selement)
        self.v = BasisFunction(self.velement)
        self.u = BasisFunction(self.velement)
        self.f = Function(self.selement)
        self.g = Function(self.selement)
        self.vf = Function(self.velement)
        self.vg = Function(self.velement)
    
    def test_1(self):
        a, b, v,  u  = self.a, self.b, self.v,  self.u
        f, g, vf, vg = self.f, self.g, self.vf, self.vg
        try:
            v*u
            self.fail()
        except (UFLException, e):
            pass
    
    def test_2(self):
        a, b, v,  u  = self.a, self.b, self.v,  self.u
        f, g, vf, vg = self.f, self.g, self.vf, self.vg
        try:
            vf*u
            self.fail()
        except (UFLException, e):
            pass
    
    def test_3(self):
        a, b, v,  u  = self.a, self.b, self.v,  self.u
        f, g, vf, vg = self.f, self.g, self.vf, self.vg
        try:
            vf*vg
            self.fail()
        except (UFLException, e):
            pass
    
    def test_4(self):
        a, b, v,  u  = self.a, self.b, self.v,  self.u
        f, g, vf, vg = self.f, self.g, self.vf, self.vg
        try:
            a+v
            self.fail()
        except (UFLException, e):
            pass

    def test_5(self):
        a, b, v,  u  = self.a, self.b, self.v,  self.u
        f, g, vf, vg = self.f, self.g, self.vf, self.vg
        try:
            vf+b
            self.fail()
        except (UFLException, e):
            pass
    
    def test_6(self):
        a, b, v,  u  = self.a, self.b, self.v,  self.u
        f, g, vf, vg = self.f, self.g, self.vf, self.vg
        tmp = vg+v+u+vf
        try:
            tmp+b
        except (UFLException, e):
            pass
    

class FormsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_source1(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element)
        try:
            a = f*v*dx
            self.fail()
        except (UFLException, e):
            pass
    
    def test_source2(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element)
        try:
            a = dot(f[0],v)
            self.fail()
        except (UFLException, e):
            pass
    
    def test_source3(self):
        element = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element)
        try:
            a = inner(f,v[0])*dx
            self.fail()
        except (UFLException, e):
            pass
    
    
    def test_mass1(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        try:
            a = u[i]*v*dx
            self.fail()
        except (UFLException, e):
            pass
    
    def test_mass2(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        try:
            a = u[i][j]
            self.fail()
        except (UFLException, e):
            pass
    
    def test_mass3(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        try:
            a = dot(u[i],v[j])*dx
            self.fail()
        except (UFLException, e):
            pass
    
    def test_mass4(self):
        element = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(u,v)*dx

    def test_duplicated_args(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        element2 = FiniteElement("Lagrange", "triangle", 2)
        v = TestFunction(element)
        u = TrialFunction(element)
        V = TestFunction(element2)
        U = TrialFunction(element2)
        a = inner(u,v)*dx + inner(V,U)*dx
        try:
            validate_form(a)
            print "FAIL:", repr(a)
            print "FAIL:", str(a)
            self.fail()
        except (UFLException, e):
            pass

    def test_duplicated_args2(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        element2 = FiniteElement("Lagrange", "triangle", 2)
        f = Function(element)
        g = Function(element2, count=f.count())
        a = (f+g)*dx
        try:
            validate_form(a)
            print "FAIL:", repr(a)
            print "FAIL:", str(a)
            self.fail()
        except (UFLException, e):
            pass

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
        M = Function(telement)
        a = inner(M*grad(u), grad(v)) * dx


    def test_navier_stokes(self):
        polygon = "triangle"
        velement = VectorElement("Lagrange", polygon, 2)
        pelement = FiniteElement("Lagrange", polygon, 1)
        TH = velement + pelement

        v, q = TestFunctions(TH)
        u, p = TrialFunctions(TH)

        f = Function(velement)
        w = Function(velement)
        Re = Constant(polygon)
        dt = Constant(polygon)

        a = dot(u, v) + dt*dot(dot(w, grad(u)), v) - dt*Re*inner(grad(u), grad(v)) + dt*dot(grad(p), v)
        L = dot(f, v)
        b = dot(u, grad(q))


if __name__ == "__main__":
    unittest.main()
