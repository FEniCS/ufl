#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-02-17 -- 2009-02-17"

import unittest
import math
        
from ufl import *
from ufl.constantvalue import as_ufl
#from ufl.classes import *

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

class DerivativeTestCase(unittest.TestCase):

    def setUp(self):
        self.cell = triangle
        self.element = FiniteElement("CG", self.cell, 1)
        self.v = TestFunction(self.element)
        self.u = TrialFunction(self.element)
        self.f = Function(self.element)
        self.g = Function(self.element)

    def testFoo(self):
        
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
