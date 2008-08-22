#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-22 -- 2008-08-22"

import unittest

from ufl import *
from ufl.algorithms import * 

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


class MassTestCase(unittest.TestCase):

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
        
        # TODO: Apply algorithms of various kinds and verify that (a, J1, J2) are equivalent, as well as (L, F).


if __name__ == "__main__":
    unittest.main()
