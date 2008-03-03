
"""
Finite element definitions

Currently, all *Element classes interit single base class,
so we can do f.ex. "if isinstance(a, UFLFiniteElement)"
"""

from polygons import *

# TODO: Finish this list
valid_families = set(("Lagrange", "CG", "DiscontinuousLagrange", "DG", "CR", "CrouzeixRaviart", "Bubble", "Nedelec", "BDM"))


class UFLFiniteElement:
    def __init__(self, polygon):
        assert polygon in valid_polygons
        self.polygon = polygon

    def __add__(self, other):
        return MixedElement(self, other)

class FiniteElement(UFLFiniteElement):
    def __init__(self, family, polygon, order):
        UFLFiniteElement.__init__(self, polygon)
        assert family in valid_families
        self.family  = family
        self.order   = order
    
    def __repr__(self):
        return "FiniteElement(%s, %s, %d)" % (repr(self.family), repr(self.polygon), self.order)

class VectorElement(UFLFiniteElement):
    def __init__(self, family, polygon, order, size=None):
        UFLFiniteElement.__init__(self, polygon)
        assert family in valid_families
        self.family  = family
        self.order   = order
        self.size    = size
    
    def __repr__(self):
        return "VectorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.size))

class TensorElement(UFLFiniteElement):
    def __init__(self, family, polygon, order, shape=None):
        UFLFiniteElement.__init__(self, polygon)
        assert family in valid_families
        self.family  = family
        self.order   = order
        self.shape   = shape
    
    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.shape))

class MixedElement(UFLFiniteElement):
    def __init__(self, *elements):
        UFLFiniteElement.__init__(self, elements[0].polygon)
        self.elements = elements

class QuadratureElement(UFLFiniteElement):
    def __init__(self, polygon, domain_type="cell"):
        UFLFiniteElement.__init__(self, polygon)
        self.domain_type = domain_type # TODO: define this better

