#!/usr/bin/env python

import unittest
from pprint import *

from ufl import *
from ufl.utilities import * 

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
        
        a = u*v*dx
        L = f*v*dx
        b = u*v*dx0 +inner(c*grad(u),grad(v))*dx1 + dot(n, grad(u))*v*ds + f*v*dx
        
        self.elements = (element,)
        self.basisfunctions = (v, u)
        self.coefficients = (c, f)
        self.forms = (a, L, b)
        
        if False:
            print 
            print [str(c) for c in self.coefficients]
            print 
            print str(self.forms[2])
            print 
            print [str(b) for b in basisfunctions(self.forms[2])]
            print 
            print self.coefficients
            print 
            print repr(self.forms[2])
            print 
            print basisfunctions(self.forms[2])
            print 
    
    def _test_flatten(self):
        element = FiniteElement("CG", "triangle", 1)
        a = Function(element, "a")
        b = Function(element, "b")
        c = Function(element, "c")
        d = Function(element, "d")
        
        a = (a+b)+(c+d)

    def _test_basisfunctions(self):
        assert self.basisfunctions == tuple(basisfunctions(self.forms[0]))
        assert tuple(self.basisfunctions[:1]) == tuple(basisfunctions(self.forms[1]))

    def test_coefficients(self):
        assert self.coefficients == tuple(coefficients(self.forms[2]))

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


suite1 = unittest.makeSuite(AlgorithmsTestCase)

allsuites = unittest.TestSuite((suite1, ))

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=0).run(allsuites)

