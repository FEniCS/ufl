#!/usr/bin/env python

from ufltestcase import UflTestCase, main

from ufl import *

#from ufl.classes import ...
from ufl.algorithms import replace

class TestTransformations(UflTestCase):

    def test_replace(self):
        V1 = FiniteElement("CG", triangle, 1)
        f1 = Coefficient(V1)
        g1 = Coefficient(V1)
        v1 = TestFunction(V1)
        u1 = TrialFunction(V1)
        a1 = f1 * g1 * u1 * v1 * dx

        V2 = FiniteElement("CG", triangle, 2)
        f2 = Coefficient(V2)
        g2 = Coefficient(V2)
        v2 = TestFunction(V2)
        u2 = TrialFunction(V2)
        a2 = f2 * g2 * u2 * v2 * dx

        mapping = {
            f1: f2,
            g1: g2,
            v1: v2,
            u1: u2,
            }
        b = replace(a1, mapping)
        self.assertEqual(b, a2)

    def test_replace_with_derivatives(self):
        V1 = FiniteElement("CG", triangle, 1)
        f1 = Coefficient(V1)
        g1 = Coefficient(V1)
        v1 = TestFunction(V1)
        u1 = TrialFunction(V1)
        a1 = u1.dx(0) * v1.dx(1) * (1 + dot(grad(f1), grad(g1))) * dx

        V2 = FiniteElement("CG", triangle, 2)
        f2 = Coefficient(V2)
        g2 = Coefficient(V2)
        v2 = TestFunction(V2)
        u2 = TrialFunction(V2)
        a2 = u2.dx(0) * v2.dx(1) * (1 + dot(0*grad(f2), grad(g2))) * dx

        mapping = {
            f1: 0, # zero!
            g1: g2,
            v1: v2,
            u1: u2,
            }
        b = replace(a1, mapping)
        self.assertEqual(b, a2)

if __name__ == "__main__":
    main()
