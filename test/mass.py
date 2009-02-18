#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-22 -- 2008-09-28"

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
        w = Function(element)
        
        f = (w**2/2)*dx
        L = w*v*dx
        a = u*v*dx
        F  = derivative(f, w, v)
        J1 = derivative(L, w, u)
        J2 = derivative(F, w, u)
        
        #self.assertTrue(F == L)
        #self.assertTrue(J1 == J2)
        #self.assertTrue(J1 == a)
        #self.assertTrue(J2 == a)
        # TODO: Apply algorithms of various kinds
        # and verify that (a, J1, J2) are equivalent,
        # as well as (L, F).


if __name__ == "__main__":
    unittest.main()
