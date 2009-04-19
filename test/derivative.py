#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-02-17 -- 2009-02-17"

import unittest
import math
        
from ufl import *
from ufl.constantvalue import as_ufl
from ufl.algorithms import expand_indices, strip_variables, post_traversal
#from ufl.classes import *

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

    def testHyperElasticity(self):
        cell = interval
        element = FiniteElement("CG", cell, 2)
        w = Function(element)
        v = TestFunction(element)
        u = TrialFunction(element)
        b = Constant(cell)
        K = Constant(cell)

        dw = w.dx(0)
        dv = v.dx(0)
        du = v.dx(0)

        E = dw + dw**2 / 2
        E = variable(E)
        Q = b*E**2
        psi = K*(exp(Q)-1)

        f = psi*dx
        F = derivative(f, w, v)
        J = derivative(F, w, u)

        f_expression = strip_variables(f.form_data().form.cell_integrals()[0].integrand())
        F_expression = strip_variables(F.form_data().form.cell_integrals()[0].integrand())
        J_expression = strip_variables(J.form_data().form.cell_integrals()[0].integrand())
        
        #classes = set(c.__class__ for c in post_traversal(f_expression))

        Kv = .2
        bv = .3
        dw = .5
        dv = .7
        du = .11
        E = dw + dw**2 / 2.
        Q = bv*E**2
        expQ = float(exp(Q))
        psi = Kv*(expQ-1)
        fv = psi
        Fv = 2*Kv*bv*E*(1+dw)*expQ*dv
        Jv = 2*Kv*bv*expQ*dv*du*(E + (1+dw)**2*(2*bv*E**2 + 1))

        def Nv(x, derivatives):
            assert derivatives == (0,)
            return dv
        
        def Nu(x, derivatives):
            assert derivatives == (0,)
            return du
        
        def Nw(x, derivatives):
            assert derivatives == (0,)
            return dw
        
        w, b, K = f.form_data().functions
        mapping = { K: Kv, b: bv, w: Nw }
        fv2 = f_expression((0,), mapping)
        self.assertAlmostEqual(fv, fv2)
        
        w, b, K = F.form_data().functions
        v, = F.form_data().basis_functions
        mapping = { K: Kv, b: bv, v: Nv, w: Nw }
        Fv2 = F_expression((0,), mapping)
        self.assertAlmostEqual(Fv, Fv2)

        w, b, K = J.form_data().functions
        v, u = J.form_data().basis_functions
        mapping = { K: Kv, b: bv, v: Nv, u: Nu, w: Nw }
        Jv2 = J_expression((0,), mapping)
        self.assertAlmostEqual(Jv, Jv2)

tests = [DerivativeTestCase]

if __name__ == "__main__":
    unittest.main()
