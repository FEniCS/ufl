#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-20 -- 2008-08-20"

import unittest

from ufl import *
from ufl.algorithms import * 

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

class ConditionalsTestCase(unittest.TestCase):

    def setUp(self):
        pass
     
    def test_lt(self):
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element)
        g = conditional(lt(f, pi), f, pi)
        a = g*v*dx

if __name__ == "__main__":
    unittest.main()
