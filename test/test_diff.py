#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-02-17 -- 2009-02-17"

from ufltestcase import UflTestCase, main
import math
        
from ufl import *
from ufl.constantvalue import as_ufl
from ufl.algorithms import expand_derivatives

class DiffTestCase(UflTestCase):

    def setUp(self):
        super(DiffTestCase, self).setUp()
        self.xv = ()
        self.vv = 5.0
        self.v = variable(self.vv)
    
    def _test(self, f, df):
        x, v = self.xv, self.v

        dfv1 = diff(f(v), v)
        dfv2 = df(v)
        dfv1 = dfv1(x)
        dfv2 = dfv2(x)
        self.assertAlmostEqual(dfv1, dfv2)
        
        dfv1 = diff(f(7*v), v)
        dfv2 = 7*df(7*v)
        dfv1 = dfv1(x)
        dfv2 = dfv2(x)
        self.assertAlmostEqual(dfv1, dfv2)

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
    
    def testExp(self):
        def f(v):  return exp(v)
        def df(v): return exp(v)
        self._test(f, df)
    
    def testLn(self):
        def f(v):  return ln(v)
        def df(v): return 1.0/v
        self._test(f, df)
    
    def testSin(self):
        def f(v):  return sin(v)
        def df(v): return cos(v)
        self._test(f, df)
    
    def testCos(self):
        def f(v):  return cos(v)
        def df(v): return -sin(v)
        self._test(f, df)

    def testTan(self):
        def f(v):  return tan(v)
        def df(v): return 2.0/(cos(2.0*v) + 1.0)
        self._test(f, df)

# TODO: Check the following tests. They run into strange math domain errors.
#     def testAsin(self):
#         def f(v):  return asin(v)
#         def df(v): return 1/sqrt(1.0 - v**2)
#         self._test(f, df)

#     def testAcos(self):
#         def f(v):  return acos(v)
#         def df(v): return -1/sqrt(1.0 - v**2)
#         self._test(f, df)

    def testAtan(self):
        def f(v):  return atan(v)
        def df(v): return 1/(1.0 + v**2)
        self._test(f, df)

    def testIndexSum(self):
        def f(v):
            # 3*v + 4*v**2 + 5*v**3
            a = as_vector((v, v**2, v**3))
            b = as_vector((3, 4, 5))
            i, = indices(1)
            return a[i]*b[i]
        def df(v): return 3 + 4*2*v + 5*3*v**2
        self._test(f, df)

    def testDiffX(self):
        cell = triangle
        x = cell.x
        f = x[0]**2 * x[1]**2
        i, = indices(1)
        df1 = diff(f, x)
        df2 = as_vector(f.dx(i), i)

        xv = (2, 3)
        df10 = df1[0](xv)
        df11 = df1[1](xv)
        df20 = df2[0](xv)
        df21 = df2[1](xv)
        self.assertAlmostEqual(df10, df20)
        self.assertAlmostEqual(df11, df21)
        self.assertAlmostEqual(df10, 2*2*9)
        self.assertAlmostEqual(df11, 2*4*3)

    # TODO: More tests involving wrapper types and indices
    
if __name__ == "__main__":
    main()
