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
        self.w = Function(self.element)
        self.xv = ()
        self.uv = 7.0
        self.vv = 13.0
        self.wv = 11.0

    def _test(self, f, df):
        x = self.xv
        u, v, w = self.u, self.v, self.w
        mapping = { v: self.vv, u: self.uv, w: self.wv }

        dfv1 = derivative(f(w), w, v)
        dfv2 = df(w, v)
        dfv1 = dfv1(x, mapping)
        dfv2 = dfv2(x, mapping)
        self.assertTrue(dfv1 == dfv2)
        
        dfv1 = derivative(f(7*w), w, v)
        dfv2 = 7*df(7*w, v)
        dfv1 = dfv1(x, mapping)
        dfv2 = dfv2(x, mapping)
        self.assertTrue(dfv1 == dfv2)

    def testFunction(self):
        def f(w):  return w
        def df(w, v): return v
        self._test(f, df)

    def testSum(self):
        def f(w):  return w + 1
        def df(w, v): return v
        self._test(f, df)

    def testProduct(self):
        def f(w):  return 3*w
        def df(w, v): return 3*v
        self._test(f, df)

    def testPower(self):
        def f(w):  return w**3
        def df(w, v): return 3*w**2*v
        self._test(f, df)
    
    def testDivision(self):
        def f(w):  return w / 3.0
        def df(w, v): return v / 3.0
        self._test(f, df)
    
    def testDivision2(self):
        def f(w):  return 3.0 / w
        def df(w, v): return -3.0 * v / w**2
        self._test(f, df)
    
    def testExp(self):
        def f(w):  return exp(w)
        def df(w, v): return v*exp(w)
        self._test(f, df)
    
    def testLn(self):
        def f(w):  return ln(w)
        def df(w, v): return v / w
        self._test(f, df)
    
    def testSin(self):
        def f(w):  return sin(w)
        def df(w, v): return v*cos(w)
        self._test(f, df)
    
    def testCos(self):
        def f(w):  return cos(w)
        def df(w, v): return -v*sin(w)
        self._test(f, df)

    def testIndexSum(self):
        def f(w):
            # 3*w + 4*w**2 + 5*w**3
            a = as_vector((w, w**2, w**3))
            b = as_vector((3, 4, 5))
            i, = indices(1)
            return a[i]*b[i]
        def df(w, v): return 3*v + 4*2*w*v + 5*3*w**2*v
        self._test(f, df)

tests = [DerivativeTestCase]

if __name__ == "__main__":
    unittest.main()
