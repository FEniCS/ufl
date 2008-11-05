#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-12 -- 2008-08-20"

# Modified by Anders Logg, 2008

import unittest
from pprint import *

from ufl import *
from ufl.algorithms import * 
from ufl.classes import Sum, Product

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

# TODO: add more tests, covering all utility algorithms

class AlgorithmsTestCase(unittest.TestCase):

    def setUp(self):
        element = FiniteElement("CG", "triangle", 1)
        
        v = TestFunction(element)
        u = TrialFunction(element)
        
        c = Function(element, "c")
        f = Function(element, "f")
        
        n = FacetNormal("triangle")
        
        a = u*v*dx
        L = f*v*dx
        b = u*v*dx(0) +inner(c*grad(u),grad(v))*dx(1) + dot(n, grad(u))*v*ds + f*v*dx
        
        self.elements = (element,)
        self.basisfunctions = (v, u)
        self.coefficients = (c, f)
        self.forms = (a, L, b)

        if False:
            print
            print
            print
            print form_info(a)
            print
            print
            print
            print form_info(L)
            print
            print
            print
            print form_info(b)
            print
            print
            print
        
        if False:
            print 
            print [str(c) for c in self.coefficients]
            print 
            print str(self.forms[2])
            print 
            print [str(b) for b in extract_basisfunctions(self.forms[2])]
            print 
            print self.coefficients
            print 
            print repr(self.forms[2])
            print 
            print extract_basisfunctions(self.forms[2])
            print 
    
    def test_flatten(self):
        element = FiniteElement("CG", "triangle", 1)
        a = Function(element, "a")
        b = Function(element, "b")
        c = Function(element, "c")
        d = Function(element, "d")
        
        a  = (a+b)+(c+d)
        fa = flatten(a)
        assert isinstance(fa, Sum) and len(fa.operands()) == 4
        assert isinstance(a,  Sum) and len(a.operands())  == 2
        aa, ab = a.operands()
        assert isinstance(aa, Sum) and len(aa.operands()) == 2
        assert isinstance(ab, Sum) and len(ab.operands()) == 2

        a  = (a*b)*(c*d)
        fa = flatten(a)
        assert isinstance(fa, Product) and len(fa.operands()) == 4
        assert isinstance(a,  Product) and len(a.operands())  == 2
        aa, ab = a.operands()
        assert isinstance(aa, Product) and len(aa.operands()) == 2
        assert isinstance(ab, Product) and len(ab.operands()) == 2

    def test_basisfunctions(self):
        assert self.basisfunctions == tuple(extract_basisfunctions(self.forms[0]))
        assert tuple(self.basisfunctions[:1]) == tuple(extract_basisfunctions(self.forms[1]))

    def test_coefficients(self):
        assert self.coefficients == tuple(extract_coefficients(self.forms[2]))

    def test_elements(self):
        #print elements(self.forms[2])
        #print unique_elements(self.forms[2])
        #print unique_classes(self.forms[2])
        d = extract_duplications(self.forms[2].cell_integrals()[0]._integrand)
        #pprint(list(d))

    def _test_walk(self):
        element = FiniteElement("CG", "triangle", 1)
        v = TestFunction(element)
        f = Function(element, "f")
        a = f*v*dx

        store = {}
        def foo(o):
            store[foo.count] = o
            foo.count += 1
        foo.count = 0

        for itg in a.cell_integrals():
            walk(itg.integrand, foo)
        logging.debug( "\n".join("%d:\t %s" % (k,v) for k,v in store.items()) )
        # TODO: test something... compare some strings perhaps.


if __name__ == "__main__":
    unittest.main()
