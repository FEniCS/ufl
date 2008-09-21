"""This module defines the Function class and a number of related
classes (functions), including Constant."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-21"

from .finiteelement import FiniteElement, VectorElement

# Modified by Anders Logg, 2008

from .base import Terminal
from .common import Counted, product
from .split import split

class Function(Terminal, Counted):
    __slots__ = ("_element", "_name")
    _globalcount = 0

    def __init__(self, element, name=None, count=None):
        Counted.__init__(self, count)
        self._element = element
        self._name = name
    
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

# TODO: Handle actual global constants?
class Constant(Function):
    __slots__ = ("_domain",)

    def __init__(self, domain, name=None, count=None):
        self._domain = domain
        element = FiniteElement("DG", domain, 0)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        if self._name is None:
            return "c_%d" % self._count
        else:
            return "c_%s" % self._name
    
    def __repr__(self):
        return "Constant(%r, %r, %r)" % (self._domain, self._name, self._count)

class VectorConstant(Function):
    __slots__ = ("_domain",)

    def __init__(self, domain, name=None, count=None): # FIXME: Size
        self._domain = domain
        element = VectorElement("DG", domain, 0)
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

    def __init__(self, domain, name=None, count=None): # FIXME: Shape and symmetries
        self._domain = domain
        element = TensorElement("DG", domain, 0)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        if self._name is None:
            return "c_%d" % self._count
        else:
            return "c_%s" % self._name
    
    def __repr__(self):
        return "TensorConstant(%r, %r, %r)" % (self._domain, self._name, self._count)

def Functions(element):
    return split(Function(element))
