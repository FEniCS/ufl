"""Form arguments defined in finite element spaces.
There are two groups: basisfunctions and coefficients,
which use the baseclasses BasisFunction and Function."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-18"


from .output import UFLException, ufl_warning
from .base import Terminal
from .finiteelement import FiniteElement, MixedElement, VectorElement
from .common import Counted


class BasisFunction(Terminal,Counted):
    __slots__ = ("_element",)
    _globalcount = 0
    def __init__(self, element, count=None):
        self._element = element
        Counted.__init__(self, count)
    
    def element(self):
        return self._element
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._element.value_shape()
    
    def __str__(self):
        return "v_%d" % self._count
    
    def __repr__(self):
        return "BasisFunction(%r, %r)" % (self._element, self._count)

def TestFunction(element):
    return BasisFunction(element, -2)

def TrialFunction(element):
    return BasisFunction(element, -1)


# FIXME: Maybe we don't need these after all:
def BasisFunctions(element):
    ufl_warning("BasisFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(BasisFunction(e) for e in element.sub_elements()) # FIXME: problem with count!

def TestFunctions(element):
    ufl_warning("BasisFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(TestFunction(e) for e in element.sub_elements()) # FIXME: problem with count!

def TrialFunctions(element):
    ufl_warning("BasisFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(TrialFunction(e) for e in element.sub_elements()) # FIXME: problem with count!


class Function(Terminal, Counted):
    __slots__ = ("_element", "_name")
    _globalcount = 0
    def __init__(self, element, name=None, count=None):
        self._element = element
        self._name = name
        Counted.__init__(self, count)
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._element.value_shape()
    
    def __str__(self):
        if self._name is None:
            return "w_%d" % self._count
        else:
            return "w_%s" % self._name
    
    def __repr__(self):
        return "Function(%r, %r, %r)" % (self._element, self._name, self._count)


class Constant(Function):
    __slots__ = ("_polygon",)

    def __init__(self, polygon, name=None, count=None):
        self._polygon = polygon
        element = FiniteElement("DG", polygon, 0)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        return "c_%d" % self._count # TODO: Use name here if available.
    
    def __repr__(self):
        return "Constant(%r, %r, %r)" % (self._polygon, self._name, self._count)

