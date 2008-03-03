#!/usr/bin/env python

"""
Finite element definitions

Currently, all *Element classes interit single base class,
so we can do f.ex. "if isinstance(a, FiniteElementBase)"
"""

from polygons import *

# TODO: Finish this list
valid_families = set(("Lagrange", "CG", "DiscontinuousLagrange", "DG", "CR", "CrouzeixRaviart", "Bubble", "Nedelec", "BDM"))


class FiniteElementBase:
    def __init__(self, polygon):
        assert polygon in valid_polygons
        self.polygon = polygon

    def __add__(self, other):
        return MixedElement(self, other)

class FiniteElement(FiniteElementBase):
    def __init__(self, family, polygon, order):
        FiniteElementBase.__init__(self, polygon)
        assert family in valid_families
        self.family  = family
        self.order   = order
    
    def __repr__(self):
        return "FiniteElement(%s, %s, %d)" % (repr(self.family), repr(self.polygon), self.order)

class VectorElement(FiniteElementBase):
    def __init__(self, family, polygon, order, size=None):
        FiniteElementBase.__init__(self, polygon)
        assert family in valid_families
        self.family  = family
        self.order   = order
        self.size    = size
    
    def __repr__(self):
        return "VectorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.size))

class TensorElement(FiniteElementBase):
    def __init__(self, family, polygon, order, shape=None):
        FiniteElementBase.__init__(self, polygon)
        assert family in valid_families
        self.family  = family
        self.order   = order
        self.shape   = shape
    
    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.shape))

class MixedElement(FiniteElementBase):
    def __init__(self, *elements):
        FiniteElementBase.__init__(self, elements[0].polygon)
        self.elements = elements

class QuadratureElement(FiniteElementBase):
    def __init__(self, polygon, domain_type="cell"):
        FiniteElementBase.__init__(self, polygon)
        self.domain_type = domain_type # TODO: define this better

