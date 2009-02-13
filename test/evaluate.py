#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-02-13 -- 2009-02-13"

import unittest

from ufl import *
from ufl.constantvalue import as_ufl
#from ufl.classes import *

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

class EvaluateTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def testScalars(self):
        s = as_ufl(123)
        e = s((5,7))
        v = 123
        self.assertTrue(e == v)

    def testZero(self):
        s = as_ufl(0)
        e = s((5,7))
        v = 0
        self.assertTrue(e == v)

    def testIdentity(self):
        cell = triangle
        I = Identity(cell.d)
        
        s = 123*I[0,0]
        e = s((5,7))
        v = 123
        self.assertTrue(e == v)
        
        s = 123*I[1,0]
        e = s((5,7))
        v = 0
        self.assertTrue(e == v)

    def testCoords(self):
        cell = triangle
        x = cell.x
        s = x[0] + x[1]
        e = s((5,7))
        v = 5 + 7
        self.assertTrue(e == v)

    def testFunction1(self):
        cell = triangle
        element = FiniteElement("CG", cell, 1)
        f = Function(element)
        s = 3*f
        e = s((5,7), { f: 123 })
        v = 3*123
        self.assertTrue(e == v)

    def testFunction2(self):
        cell = triangle
        element = FiniteElement("CG", cell, 1)
        f = Function(element)
        def g(x):
            return x[0]
        s = 3*f
        e = s((5,7), { f: g })
        v = 3*5
        self.assertTrue(e == v)

    def testBasisFunction2(self):
        cell = triangle
        element = FiniteElement("CG", cell, 1)
        f = BasisFunction(element)
        def g(x):
            return x[0]
        s = 3*f
        e = s((5,7), { f: g })
        v = 3*5
        self.assertTrue(e == v)

    def testAlgebra(self):
        cell = triangle
        x = cell.x
        s = 3*(x[0] + x[1]) - 7 + x[0]**(x[1]/2)
        e = s((5,7))
        v = 3*(5. + 7.) - 7 + 5.**(7./2)
        self.assertTrue(e == v)

    def testIndexSum(self):
        cell = triangle
        x = cell.x
        i, = indices(1)
        s = x[i]*x[i]
        e = s((5,7))
        v = 5**2 + 7**2
        self.assertTrue(e == v)
    
    def testIndexSum2(self):
        cell = triangle
        x = cell.x
        I = Identity(cell.d)
        i, j = indices(2)
        s = (x[i]*x[j])*I[i,j]
        e = s((5,7))
        #v = sum_i sum_j x_i x_j delta_ij = x_0 x_0 + x_1 x_1
        v = 5**2 + 7**2
        self.assertTrue(e == v)


if __name__ == "__main__":
    unittest.main()
