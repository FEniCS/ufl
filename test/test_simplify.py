#!/usr/bin/env python

from ufltestcase import UflTestCase, main
from ufl.classes import Sum, Product

from ufl import *

class ElementsTestCase(UflTestCase):

    def test_products(self):
        element = FiniteElement("CG", "triangle", 1)
        f = Coefficient(element)
        g = Coefficient(element)
        
        # Test simplification of basic multiplication
        a = f
        b = 1*f
        self.assertEqual(a, b)
        
        # Test simplification of self-multiplication
        a = f*f
        b = f**2
        self.assertEqual(a, b)
        
        # Test simplification of flattened self-multiplication (may occur in algorithms)
        a = Product(f,f,f)
        b = f**3
        self.assertEqual(a, b)
        
        # Test simplification of flattened self-multiplication (may occur in algorithms)
        a = Product(f,f,f,f)
        b = f**4
        self.assertEqual(a, b)
        
    def test_sums(self):
        element = FiniteElement("CG", "triangle", 1)
        f = Coefficient(element)
        g = Coefficient(element)
        
        # Test collapsing of basic sum
        a = f + f
        b = 2*f
        self.assertEqual(a, b)
        
        # Test collapsing of flattened sum (may occur in algorithms)
        a = Sum(f, f, f)
        b = 3*f
        self.assertEqual(a, b)
        a = Sum(f, f, f, f)
        b = 4*f
        self.assertEqual(a, b)
        
        # Test reordering of operands
        a = f + g
        b = g + f
        self.assertEqual(a, b)
        
        # Test reordering of operands and collapsing sum
        a = f + g + f # not collapsed, but ordered
        b = g + f + f # not collapsed, but ordered
        c = (g + f) + f # not collapsed, but ordered
        d = f + (f + g) # not collapsed, but ordered
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)
        
        # Test reordering of operands and collapsing sum
        a = f + f + g # collapsed
        b = g + (f + f) # collapsed
        self.assertEqual(a, b)

if __name__ == "__main__":
    main()
