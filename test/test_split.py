#!/usr/bin/env python

from ufl import *

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-03-14 -- 2009-03-14"

from ufltestcase import UflTestCase, main

from ufl import *

class SplitTestCase(UflTestCase):

    def test_split(self):
        cell = triangle
        d = cell.geometric_dimension()
        f = FiniteElement("CG", cell, 1)
        v = VectorElement("CG", cell, 1)
        w = VectorElement("CG", cell, 1, dim=d+1)
        t = TensorElement("CG", cell, 1)
        s = TensorElement("CG", cell, 1, symmetry=True)
        r = TensorElement("CG", cell, 1, symmetry={(1, 0): (0, 1)}, shape=(d, d))
        m = MixedElement(f, v, w, t, s, r)

        # Shapes of all these functions are correct:
        self.assertEqual((), Coefficient(f).shape())
        self.assertEqual((d,), Coefficient(v).shape())
        self.assertEqual((d+1,), Coefficient(w).shape())
        self.assertEqual((d, d), Coefficient(t).shape())
        self.assertEqual((d, d), Coefficient(s).shape())
        self.assertEqual((d, d), Coefficient(r).shape())
        self.assertEqual((3*d*d + 2*d + 2,), Coefficient(m).shape()) # sum of value sizes, not accounting for symmetries

        # Shapes of subelements are reproduced:
        g = Coefficient(m)
        s, = g.shape()
        for g2 in split(g):
            s -= product(g2.shape())
        self.assertEqual(s, 0)
        
        # TODO: Should functions on mixed elements (vector+vector) be able to have tensor shape instead of vector shape? Think Marie wants this for BDM+BDM?
        v2 = MixedElement(v, v)
        m2 = MixedElement(t, t)
        #self.assertEqual(d, 2)
        #self.assertEqual((2,2), Coefficient(v2).shape())
        self.assertEqual((d+d,), Coefficient(v2).shape())
        self.assertEqual((2*d*d,), Coefficient(m2).shape())

if __name__ == "__main__":
    main()
