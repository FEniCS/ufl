#!/usr/bin/env python

from ufltestcase import UflTestCase, main

from ufl import *
from ufl.algorithms import *

# TODO: these tests only verify that the syntax is possible, how much more can we test without a form compiler?

# TODO: add more forms, covering all UFL operators


class IllegalExpressionsTestCase(UflTestCase):
    def setUp(self):
        super(IllegalExpressionsTestCase, self).setUp()
        self.selement = FiniteElement("Lagrange", "triangle", 1)
        self.velement = VectorElement("Lagrange", "triangle", 1)
        self.a = Argument(self.selement)
        self.b = Argument(self.selement)
        self.v = Argument(self.velement)
        self.u = Argument(self.velement)
        self.f = Coefficient(self.selement)
        self.g = Coefficient(self.selement)
        self.vf = Coefficient(self.velement)
        self.vg = Coefficient(self.velement)

    def tearDown(self):
        super(IllegalExpressionsTestCase, self).tearDown()

    def test_mul_v_u(self):
        self.assertRaises(UFLException, lambda: self.v * self.u)

    def test_mul_vf_u(self):
        self.assertRaises(UFLException, lambda: self.vf * self.u)

    def test_mul_vf_vg(self):
        self.assertRaises(UFLException, lambda: self.vf * self.vg)

    def test_add_a_v(self):
        self.assertRaises(UFLException, lambda: self.a + self.v)

    def test_add_vf_b(self):
        self.assertRaises(UFLException, lambda: self.vf + self.b)

    def test_add_vectorexpr_b(self):
        tmp = self.vg + self.v + self.u + self.vf
        self.assertRaises(UFLException, lambda: tmp + self.b)

class FormsTestCase(UflTestCase):

    def setUp(self):
        super(FormsTestCase, self).setUp()

    def tearDown(self):
        super(FormsTestCase, self).tearDown()

    def test_source1(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Coefficient(element)
        self.assertRaises(UFLException, lambda: f*v*dx)

    def test_source2(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Coefficient(element)
        self.assertRaises(UFLException, lambda: dot(f[0], v))

    def test_source3(self):
        element = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Coefficient(element)
        self.assertRaises(UFLException, lambda: inner(f, v[0])*dx)

    def test_mass1(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        self.assertRaises(UFLException, lambda: u[i]*v*dx)

    def test_mass2(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        self.assertRaises(UFLException, lambda: u[i][j])

    def test_mass3(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        self.assertRaises(UFLException, lambda: dot(u[i], v[j])*dx)

    def test_mass4(self):
        element = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(u,v)*dx
        # TODO: Assert something? What are we testing here?

    def check_validate_raises(self, a):
        def store_if_nothrow():
            validate_form(a)
            store_if_nothrow.nothrow = True
        store_if_nothrow.nothrow = False

        self.assertRaises(UFLException, store_if_nothrow)

        if store_if_nothrow.nothrow:
            print "in check_validate_raises:"
            print "repr =", repr(a)
            print "str =", str(a)

    def test_duplicated_args(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        element2 = FiniteElement("Lagrange", "triangle", 2)
        v = TestFunction(element)
        u = TrialFunction(element)
        V = TestFunction(element2)
        U = TrialFunction(element2)
        a = inner(u,v)*dx + inner(V,U)*dx
        self.check_validate_raises(a)

    def test_duplicated_args2(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        element2 = FiniteElement("Lagrange", "triangle", 2)
        f = Coefficient(element)
        g = Coefficient(element2, count=f.count())
        a = (f+g)*dx
        self.check_validate_raises(a)

    def test_stiffness1(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = dot(grad(u), grad(v)) * dx
        # TODO: Assert something? What are we testing here?

    def test_stiffness2(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(grad(u), grad(v)) * dx
        # TODO: Assert something? What are we testing here?

    def test_stiffness3(self):
        element = VectorElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(grad(u), grad(v)) * dx
        # TODO: Assert something? What are we testing here?

    def test_stiffness_with_conductivity(self):
        velement = VectorElement("Lagrange", "triangle", 1)
        telement = TensorElement("Lagrange", "triangle", 1)
        v = TestFunction(velement)
        u = TrialFunction(velement)
        M = Coefficient(telement)
        a = inner(M*grad(u), grad(v)) * dx
        # TODO: Assert something? What are we testing here?

    def test_navier_stokes(self):
        polygon = "triangle"
        velement = VectorElement("Lagrange", polygon, 2)
        pelement = FiniteElement("Lagrange", polygon, 1)
        TH = velement * pelement

        v, q = TestFunctions(TH)
        u, p = TrialFunctions(TH)

        f = Coefficient(velement)
        w = Coefficient(velement)
        Re = Constant(polygon)
        dt = Constant(polygon)

        a = dot(u, v) + dt*dot(dot(w, grad(u)), v) - dt*Re*inner(grad(u), grad(v)) + dt*dot(grad(p), v)
        L = dot(f, v)
        b = dot(u, grad(q))

        # TODO: Assert something? What are we testing here?

if __name__ == "__main__":
    main()
