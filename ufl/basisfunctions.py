"""Form arguments defined in finite element spaces.
There are two groups: basisfunctions and coefficients,
which use the baseclasses BasisFunction and Function."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-20"


from .output import ufl_warning, ufl_error
from .base import Terminal
from .finiteelement import FiniteElement, MixedElement, VectorElement
from .common import Counted


class BasisFunction(Terminal, Counted):
    __slots__ = ("_element", "_parent")
    _globalcount = 0
    def __init__(self, element, parent=None, count=None):
        Counted.__init__(self, count)
        self._element = element
        self._parent = parent
    
    def element(self):
        return self._element
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._element.value_shape()
    
    def __str__(self):
        return "v_%d" % self._count
    
    def __repr__(self):
        return "BasisFunction(%r, %r %r)" % (self._element, self._parent, self._count)

def TestFunction(element):
    return BasisFunction(element, None, -2)

def TrialFunction(element):
    return BasisFunction(element, None, -1)


# FIXME: Maybe we don't need these after all?
def BasisFunctions(element):
    ufl_warning("BasisFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        ufl_error("Expecting MixedElement instance.")
    bf = BasisFunction(element)
    return tuple(BasisFunction(e, parent=bf, count=bf._count) for e in element.sub_elements()) # FIXME: What should count be?

def TestFunctions(element):
    ufl_warning("TestFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        ufl_error("Expecting MixedElement instance.")
    bf = TestFunction(element)
    return tuple(BasisFunction(e, parent=bf, count=bf._count) for e in element.sub_elements()) # FIXME: What should count be?

def TrialFunctions(element):
    ufl_warning("TrialFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        ufl_error("Expecting MixedElement instance.")
    bf = TrialFunction(element)
    return tuple(BasisFunction(e, parent=bf, count=bf._count) for e in element.sub_elements()) # FIXME: What should count be?


class Function(Terminal, Counted):
    __slots__ = ("_element", "_name", "_parent")
    _globalcount = 0
    def __init__(self, element, name=None, parent=None, count=None):
        Counted.__init__(self, count)
        self._element = element
        self._name = name
        self._parent = parent
    
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


# FIXME: Maybe we don't need these after all? 
def Functions(element):
    ufl_warning("Functions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        ufl_error("Expecting MixedElement instance.")
    f = Function(element)
    return tuple(Function(e, parent=f, count=f._count) for e in element.sub_elements()) # FIXME: What should count be?

class Constant(Function):
    __slots__ = ("_polygon",)

    def __init__(self, polygon, name=None, count=None):
        self._polygon = polygon
        element = FiniteElement("DG", polygon, 0)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        if self._name is None:
            return "c_%d" % self._count
        else:
            return "c_%s" % self._name
    
    def __repr__(self):
        return "Constant(%r, %r, %r)" % (self._polygon, self._name, self._count)
