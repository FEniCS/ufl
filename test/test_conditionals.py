#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-20 -- 2008-08-20"

from ufltestcase import UflTestCase, main

from ufl import *
from ufl.algorithms import * 

class ConditionalsTestCase(UflTestCase):

    def test_lt(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Coefficient(element)
        g = conditional(lt(f, pi), f, pi)
        a = g*v*dx

if __name__ == "__main__":
    main()
