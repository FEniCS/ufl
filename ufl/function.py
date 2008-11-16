"""This module defines the Function class and a number 
of related classes (functions), including Constant."""


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-31"

# Modified by Anders Logg, 2008

from ufl.finiteelement import FiniteElement, VectorElement, TensorElement
from ufl.base import Terminal
from ufl.common import Counted, product
from ufl.split import split

# --- The Function class represents a coefficient function argument to a form ---

class Function(Terminal, Counted):
    __slots__ = ("_element", "_name")
    _globalcount = 0

    def __init__(self, element, name=None, count=None):
        Counted.__init__(self, count)
        self._element = element
        self._name = name
    
    def element(self):
        return self._element
    
    def shape(self):
        return self._element.value_shape()
    
    def domain(self):
        return self._element.domain()
    
    def __str__(self):
        if self._name is None:
            count = str(self._count)
            if len(count) == 1:
                return "w_%s" % count
            else:
                return "w_{%s}" % count
        else:
            return "w_%s" % self._name
    
    def __repr__(self):
        return "Function(%r, %r, %r)" % (self._element, self._name, self._count)

    def __eq__(self, other):
        return isinstance(other, Function) and self._count == other._count

# --- Subclasses for defining constant functions without specifying element ---

# TODO: Handle actual global constants?

class Constant(Function):
    __slots__ = ("_domain",)

    def __init__(self, domain, name=None, count=None):
        self._domain = domain
        element = FiniteElement("DG", domain, 0)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        if self._name is None:
            count = str(self._count)
            if len(count) == 1:
                return "c_%s" % count
            else:
                return "c_{%s}" % count
        else:
            return "c_%s" % self._name
    
    def __repr__(self):
        return "Constant(%r, %r, %r)" % (self._domain, self._name, self._count)

class VectorConstant(Function):
    __slots__ = ("_domain",)

    def __init__(self, domain, dim=None, name=None, count=None):
        self._domain = domain
        element = VectorElement("DG", domain, 0, dim=dim)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        if self._name is None:
            return "c_%d" % self._count
        else:
            return "c_%s" % self._name
    
    def __repr__(self):
        return "VectorConstant(%r, %r, %r)" % (self._domain, self._name, self._count)

class TensorConstant(Function):
    __slots__ = ("_domain",)

    def __init__(self, domain, shape=None, symmetry=None, name=None, count=None):
        self._domain = domain
        element = TensorElement("DG", domain, 0, shape=shape, symmetry=symmetry)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        if self._name is None:
            return "c_%d" % self._count
        else:
            return "c_%s" % self._name
    
    def __repr__(self):
        return "TensorConstant(%r, %r, %r)" % (self._domain, self._name, self._count)

# --- Helper functions for subfunctions on mixed elements ---

def Functions(element):
    return split(Function(element))
