#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-22 -- 2008-08-22"

import unittest

from ufl import *
from ufl.algorithms import * 
from ufl.algorithms.swiginac import *

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


class SwiginacTestCase(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_something(self):
        element = FiniteElement("CG", "triangle", 1)
        
        v = TestFunction(element)
        u = TrialFunction(element)
        w = Function(element, "w")
        
        f = (w**2/2)*dx
        L = w*v*dx
        a = u*v*dx
        F  = Derivative(f, w)
        J1 = Derivative(L, w)
        J2 = Derivative(F, w)

    def test_number(self):
        context = None
        f = Number(1.23)
        g = evaluate_as_swiginac(f, context)
        self.assertTrue((g-1.23) == 0)


if __name__ == "__main__":
    unittest.main()
