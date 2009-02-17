#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-02-17 -- 2009-02-17"

import unittest
import math
        
from ufl import *
from ufl.constantvalue import as_ufl
#from ufl.classes import *
from ufl.algorithms import expand_derivatives

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

class DiffTestCase(unittest.TestCase):

    def setUp(self):
        self.xv = ()
        self.vv = 5.0
        self.v = variable(self.vv)

    def _test(self, f, df):
        x, v = self.xv, self.v

        dfv1 = diff(f(v), v)
        dfv2 = df(v)
        dfv1 = dfv1(x)
        dfv2 = dfv2(x)
        self.assertTrue(dfv1 == dfv2)
        
        dfv1 = diff(f(7*v), v)(x)
        dfv2 = 7*df(v)(x)
        dfv1 = dfv1(x)
        dfv2 = dfv2(x)
        self.assertTrue(dfv1 == dfv2)

    def testVariable(self):
        def f(v):  return v
        def df(v): return as_ufl(1)
        self._test(f, df)

    def testSum(self):
        def f(v):  return v + 1
        def df(v): return as_ufl(1)
        self._test(f, df)

    def testProduct(self):
        def f(v):  return 3*v
        def df(v): return as_ufl(3)
        self._test(f, df)

    def testPower(self):
        def f(v):  return v**3
        def df(v): return 3*v**2
        self._test(f, df)
    
    def testDivision(self):
        def f(v):  return v / 3.0
        def df(v): return as_ufl(1.0/3.0)
        self._test(f, df)
    
    def testDivision2(self):
        def f(v):  return 3.0 / v
        def df(v): return -3.0 / v**2
        self._test(f, df)

if __name__ == "__main__":
    unittest.main()
