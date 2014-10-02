#!/usr/bin/env py.test

from ufl import *

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-03-14 -- 2009-03-14"

import pytest

from ufl import *

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
    assert () == Coefficient(f).ufl_shape
    self.assertEqual((d,), Coefficient(v).ufl_shape)
    self.assertEqual((d+1,), Coefficient(w).ufl_shape)
    self.assertEqual((d, d), Coefficient(t).ufl_shape)
    self.assertEqual((d, d), Coefficient(s).ufl_shape)
    self.assertEqual((d, d), Coefficient(r).ufl_shape)
    self.assertEqual((3*d*d + 2*d + 2,), Coefficient(m).ufl_shape) # sum of value sizes, not accounting for symmetries

    # Shapes of subelements are reproduced:
    g = Coefficient(m)
    s, = g.ufl_shape
    for g2 in split(g):
        s -= product(g2.ufl_shape)
    assert s == 0

    # TODO: Should functions on mixed elements (vector+vector) be able to have tensor shape instead of vector shape? Think Marie wants this for BDM+BDM?
    v2 = MixedElement(v, v)
    m2 = MixedElement(t, t)
    #assert d == 2
    #self.assertEqual((2,2), Coefficient(v2).ufl_shape)
    self.assertEqual((d+d,), Coefficient(v2).ufl_shape)
    self.assertEqual((2*d*d,), Coefficient(m2).ufl_shape)
