#!/usr/bin/env python

__authors__ = "Automatically generated from .tex files"
__date__ = "2009-02-07 -- 2009-02-07"

import unittest

from ufl import *
from ufl.classes import *
from ufl.algorithms import * 

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)

class ManualTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_0(self):
        element = FiniteElement("CG", triangle, 1)
        element = FiniteElement("DG", tetrahedron, 0)
        

if __name__ == "__main__":
    unittest.main()
