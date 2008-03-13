#!/usr/bin/env python

"""
Finite element definitions

Currently, all *Element classes interit single base class,
so we can do f.ex. "if isinstance(a, FiniteElementBase)"
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 13th 2008"

from shapes import *

# TODO: Finish this list
valid_families = set(("Lagrange", "CG", "DiscontinuousLagrange", "DG", "CR", "CrouzeixRaviart", "Bubble", "Nedelec", "BDM"))


# FIXME: Some elements are vector-valued (BDM, RT, BDFM, Nedelec)
# FIXME: Why do we need an extra class FiniteElementBase, should be enough with FiniteElement

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
        self.value_rank = 0
    
    def __repr__(self):
        return "FiniteElement(%s, %s, %d)" % (repr(self.family), repr(self.polygon), self.order)

class VectorElement(FiniteElementBase):
    def __init__(self, family, polygon, order, size=None):
        FiniteElementBase.__init__(self, polygon)
        assert family in valid_families
        self.family  = family
        self.order   = order
        self.size    = size
        self.value_rank = 1
    
    def __repr__(self):
        return "VectorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.size))

class TensorElement(FiniteElementBase):
    def __init__(self, family, polygon, order, shape=None):
        FiniteElementBase.__init__(self, polygon)
        assert family in valid_families
        self.family  = family
        self.order   = order
        self.shape   = shape
        self.value_rank = 2
    
    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.shape))

class MixedElement(FiniteElementBase):
    def __init__(self, *elements):
        FiniteElementBase.__init__(self, elements[0].polygon)
        ufl_assert(all(e.polygon == elements[0].polygon for e in elements))
        self.elements = elements

class QuadratureElement(FiniteElementBase):
    def __init__(self, polygon, domain_type="cell"):
        FiniteElementBase.__init__(self, polygon)
        self.domain_type = domain_type # TODO: define this better

