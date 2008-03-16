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
        pass
    
    def test_walk(self):
        element = FiniteElement("Lagrange", "triangle", 1)
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

    def test_flatten(self):
        pass



suite1 = unittest.makeSuite(AlgorithmsTestCase)

allsuites = unittest.TestSuite((suite1, ))

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=0).run(allsuites)

