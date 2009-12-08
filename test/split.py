
from ufl import *


#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-03-14 -- 2009-03-14"

import unittest

from ufl import *

class SplitTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_split(self):
        cell = triangle
        d = cell.d
        f = FiniteElement("CG", cell, 1)
        v = VectorElement("CG", cell, 1)
        w = VectorElement("CG", cell, 1, dim=d+1)
        t = TensorElement("CG", cell, 1)
        s = TensorElement("CG", cell, 1, symmetry=True)
        r = TensorElement("CG", cell, 1, symmetry={(1,0): (0,1)}, shape=(d,d))
        m = MixedElement(f, v, w, t, s, r)

        # Shapes of all these functions are correct:
        self.assertTrue(() == Coefficient(f).shape())
        self.assertTrue((d,) == Coefficient(v).shape())
        self.assertTrue((d+1,) == Coefficient(w).shape())
        self.assertTrue((d,d) == Coefficient(t).shape())
        self.assertTrue((d,d) == Coefficient(s).shape())
        self.assertTrue((d,d) == Coefficient(r).shape())
        self.assertTrue((3*d*d + 2*d + 2,) == Coefficient(m).shape()) # sum of value sizes, not accounting for symmetries

        # Shapes of subelements are reproduced:
        g = Coefficient(m)
        s, = g.shape()
        for g2 in split(g):
            s -= product(g2.shape())
        self.assertTrue(s == 0)
        
        # TODO: Should functions on mixed elements (vector+vector) be able to have tensor shape instead of vector shape? Think Marie wants this for BDM+BDM?
        v2 = MixedElement(v, v)
        m2 = MixedElement(t, t)
        #self.assertTrue(d == 2 and (2,2) == Coefficient(v2).shape())
        self.assertTrue((d+d,) == Coefficient(v2).shape())
        self.assertTrue((2*d*d,) == Coefficient(m2).shape())
    
tests = [SplitTestCase]

if __name__ == "__main__":
    unittest.main()
