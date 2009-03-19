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
        pass
    
    def test_expand_indices(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        velement = VectorElement("Lagrange", cell, 1)
        telement = TensorElement("Lagrange", cell, 1)
        sf = Function(element)
        vf = Function(velement)
        tf = Function(velement)
        
        def SF(x):
            return 3
        
        def VF(x):
            return (5, 7)
        
        def TF(x):
            return ((11, 13), (17, 19))
        
        def compare(f, value):
            g1 = expand_derivatives(f)
            g2 = expand_indices(g1)
            
            x = (1.23, 3.14)
            mapping = { sf: SF, vf: VF, tf: TF }
            
            g1v = g1(x, mapping)
            g2v = g2(x, mapping)
            
            self.assertAlmostEqual( g1v, value )
            self.assertAlmostEqual( g2v, value )
        
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
