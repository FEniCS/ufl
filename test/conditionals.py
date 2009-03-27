#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-20 -- 2008-08-20"

import unittest

from ufl import *
from ufl.algorithms import * 

class ConditionalsTestCase(unittest.TestCase):

    def setUp(self):
        pass
     
    def test_lt(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element)
        g = conditional(lt(f, pi), f, pi)
        a = g*v*dx

tests = [ConditionalsTestCase]

if __name__ == "__main__":
    unittest.main()
