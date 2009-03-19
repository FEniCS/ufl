#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-03-19 -- 2009-03-19"

# Modified by Anders Logg, 2008

import unittest
import math
from pprint import *

from ufl import *
from ufl.algorithms import * 
from ufl.classes import Sum, Product

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

# TODO: add more tests, covering all utility algorithms

class ExpandIndicesTestCase(unittest.TestCase):

    def setUp(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        velement = VectorElement("Lagrange", cell, 1)
        telement = TensorElement("Lagrange", cell, 1)
        self.sf = Function(element)
        self.vf = Function(velement)
        self.tf = Function(telement)
        
        def SF(x, derivatives=()):
            if derivatives == ():
                return 3
            elif derivatives == (0,):
                return 0.3
            elif derivatives == (1,):
                return 0.3
            return 0
        
        def VF(x, derivatives=()):
            if derivatives == ():
                return (5, 7)
            elif derivatives == (0,):
                return (0.5, 0.7)
            elif derivatives == (1,):
                return (0.5, 0.7)
            return (0, 0)
        
        def TF(x, derivatives=()):
            if derivatives == ():
                return ((11, 13), (17, 19))
            elif derivatives == (0,):
                return ((1.1, 1.3), (1.7, 1.9))
            elif derivatives == (1,):
                return ((1.1, 1.3), (1.7, 1.9))
            return ((0, 0), (0, 0))
        
        self.x = (1.23, 3.14)
        self.mapping = { self.sf: SF, self.vf: VF, self.tf: TF }
        
    def compare(self, f, value):
        g1 = expand_derivatives(f)
        g2 = expand_indices(g1)
        
        g1v = g1(self.x, self.mapping)
        g2v = g2(self.x, self.mapping)
        
        self.assertAlmostEqual( g1v, value )
        self.assertAlmostEqual( g2v, value )

    def test_basic_expand_indices(self):
        sf = self.sf
        vf = self.vf
        tf = self.tf
        compare = self.compare

        # Simple expressions with no indices or derivatives to expand
        compare(sf, 3)
        compare(sf + 1, 4)
        compare(sf - 2.5, 0.5)
        compare(sf/2, 1.5)
        compare(sf/0.5, 6)
        compare(sf**2, 9)
        compare(sf**0.5, 3**0.5)
        compare(sf**3, 27)
        compare(0.5**sf, 0.5**3)
        compare(sf * (sf/6), 1.5)
        compare(sin(sf), math.sin(3))
        compare(cos(sf), math.cos(3))
        compare(exp(sf), math.exp(3))
        compare(ln(sf), math.log(3))

        # Simple indexing
        compare(vf[0], 5)
        compare(vf[0] + 1, 6)
        compare(vf[0] - 2.5, 2.5)
        compare(vf[0]/2, 2.5)
        compare(vf[0]/0.5, 10)
        compare(vf[0]**2, 25)
        compare(vf[0]**0.5, 5**0.5)
        compare(vf[0]**3, 125)
        compare(0.5**vf[0], 0.5**5)
        compare(vf[0] * (vf[0]/6), 5*(5./6))
        compare(sin(vf[0]), math.sin(5))
        compare(cos(vf[0]), math.cos(5))
        compare(exp(vf[0]), math.exp(5))
        compare(ln(vf[0]), math.log(5))
        
        # Double indexing
        compare(tf[1,1], 19)
        compare(tf[1,1] + 1, 20)
        compare(tf[1,1] - 2.5, 16.5)
        compare(tf[1,1]/2, 9.5)
        compare(tf[1,1]/0.5, 38)
        compare(tf[1,1]**2, 19**2)
        compare(tf[1,1]**0.5, 19**0.5)
        compare(tf[1,1]**3, 19**3)
        compare(0.5**tf[1,1], 0.5**19)
        compare(tf[1,1] * (tf[1,1]/6), 19*(19./6))
        compare(sin(tf[1,1]), math.sin(19))
        compare(cos(tf[1,1]), math.cos(19))
        compare(exp(tf[1,1]), math.exp(19))
        compare(ln(tf[1,1]), math.log(19))

    def test_expand_indices_index_sum(self):
        sf = self.sf
        vf = self.vf
        tf = self.tf
        compare = self.compare

        # Basic index sums
        compare(vf[i]*vf[i], 5*5+7*7)
        compare(vf[j]*tf[i,j]*vf[i], 5*5*11 + 5*7*13 + 5*7*17 + 7*7*19)
        compare(vf[j]*tf.T[j,i]*vf[i], 5*5*11 + 5*7*13 + 5*7*17 + 7*7*19)
        compare(tf[i,i], 11 + 19)
        compare(tf[i,j]*(tf[j,i]+outer(vf, vf)[i,j]), (5*5+11)*11 + (7*5+17)*13 + (7*5+13)*17 + (7*7+19)*19)
        compare( as_tensor( as_tensor(tf[i,j], (i,j))[k,l], (l,k) )[i,i], 11 + 19 )
    
    def test_expand_indices_derivatives(self):
        sf = self.sf
        vf = self.vf
        tf = self.tf
        compare = self.compare

        # Basic derivatives
        compare(sf.dx(0), 0.3)
        compare(sf.dx(1), 0.3)
        compare(sf.dx(i)*vf[i], 0.3*5 + 0.3*7)

    def _test_expand_indices2(self): # Derivatives are currently a problem
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        f = Function(element)
        v = TestFunction(element)
        u = TrialFunction(element)

        a = div(grad(v))*u*dx
        #a1 = evaluate(a)
        a = expand_derivatives(a)
        #a2 = evaluate(a)
        a = expand_indices(a)
        #a3 = evaluate(a) # TODO: How to define values of derivatives?
        # TODO: Compare a1, a2, a3
        # TODO: Test something more

tests = [ExpandIndicesTestCase]

if __name__ == "__main__":
    unittest.main()
