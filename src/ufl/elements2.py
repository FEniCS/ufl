#!/usr/bin/env python

"""
Finite element classes.

All finite element classes inherit the base class FiniteElement.
The properties they have in common are:
- polygon
- value_shape
- value_rank

The subclasses of FiniteElement are:
- MixedElement, which is a combination of any list of other elements
and
- ScalarElement
- VectorElement
- TensorElement
which all have the properties family and order.
UFL does not check if family and order are valid
in combination with the chosen polygon and value shape,
it is up to the form compiler to check f.ex. that
family="Nedelec" isn't used as a ScalarElement or TensorElement.

VectorElement takes an optional parameter size for nonstandard vector sizes,
and TensorElement takes the parameters shape and symmetric.
"""


__authors__ = "Martin Sandve Alnes"
__date__ = "March 13th 2008"

from ufl_io import *
from shapes import *


class FiniteElement:
    def __init__(self, polygon, value_rank, value_shape):
        ufl_assert(polygon in valid_polygons, "Invalid polygon")
        self.polygon     = polygon
        
        ufl_assert(isinstance(value_shape, tuple), "")
        self.value_shape = value_shape
        self.value_rank  = len(value_shape)
    
    def __add__(self, other):
        return MixedElement(self, other)


class MixedElement(FiniteElement):
    def __init__(self, *elements):
        ufl_assert(all(e.polygon == elements[0].polygon for e in elements), "Polygon mismatch in elements.")
        
        value_rank  = None # FIXME: get from elements
        value_shape = None # FIXME: get from elements
        
        FiniteElement.__init__(self, elements[0].polygon, value_rank, value_shape)
        
        self.elements = elements


class ScalarElement(FiniteElement):
    def __init__(self, family, polygon, order):
        FiniteElement.__init__(self, polygon, 0, tuple())
        self.family = family
        self.order  = order
    
    def __repr__(self):
        return "ScalarElement(%s, %s, %d)" % (repr(self.family), repr(self.polygon), self.order)


class VectorElement(FiniteElement):
    def __init__(self, family, polygon, order, size=None):
        FiniteElement.__init__(self, polygon, 1, tuple(size))
        self.family = family
        self.order  = order
    
    def __repr__(self):
        return "VectorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.size))


class TensorElement(FiniteElement):
    def __init__(self, family, polygon, order, shape=None, symmetric=False):
        value_rank = 2 if shape is None else len(shape)
        FiniteElement.__init__(self, polygon, value_rank, shape)
        self.family = family
        self.order  = order
        self.symmetric = symmetric
    
    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.shape), repr(self.symmetric))

