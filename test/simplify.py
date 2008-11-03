#!/usr/bin/env python

import unittest
from ufl.classes import Sum, Product

from ufl import *

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


class ElementsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_products(self):
        element = FiniteElement("CG", "triangle", 1)
        f = Function(element)
        g = Function(element)
        
        # Test simplification of basic multiplication
        a = f
        b = 1*f
        self.assertTrue(a == b)
        
        # Test simplification of self-multiplication
        a = f*f
        b = f**2
        self.assertTrue(a == b)
        
        # Test simplification of flattened self-multiplication (may occur in algorithms)
        a = Product(f,f,f)
        b = f**3
        self.assertTrue(a == b)
        
        # Test simplification of flattened self-multiplication (may occur in algorithms)
        a = Product(f,f,f,f)
        b = f**4
        self.assertTrue(a == b)
        
    def test_sums(self):
        element = FiniteElement("CG", "triangle", 1)
        f = Function(element)
        g = Function(element)
        
        # Test collapsing of basic sum
        a = f + f
        b = 2*f
        self.assertTrue(a == b)
        
        # Test collapsing of flattened sum (may occur in algorithms)
        a = Sum(f, f, f)
        b = 3*f
        self.assertTrue(a == b)
        a = Sum(f, f, f, f)
        b = 4*f
        self.assertTrue(a == b)
        
        # Test reordering of operands
        a = f + g
        b = g + f
        self.assertTrue(a == b)
        
        # Test reordering of operands and collapsing sum
        a = f + g + f # not collapsed, but ordered
        b = g + f + f # not collapsed, but ordered
        c = (g + f) + f # not collapsed, but ordered
        d = f + (f + g) # not collapsed, but ordered
        self.assertTrue(a == b)
        self.assertTrue(a == c)
        self.assertTrue(a == d)
        
        # Test reordering of operands and collapsing sum
        a = f + f + g # collapsed
        b = g + (f + f) # collapsed
        self.assertTrue(a == b)

if __name__ == "__main__":
    unittest.main()
