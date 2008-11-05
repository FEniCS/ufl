"""This module defines the class BasisFunction and a number of related
classes (functions), including TestFunction and TrialFunction."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-11-05"

# Modified by Anders Logg, 2008

from .base import Terminal
from .common import Counted, product
from .split import split

# --- Class representing a basis function argument in a form ---

class BasisFunction(Terminal, Counted):
    __slots__ = ("_element", "_repr")
    _globalcount = 0

    def __init__(self, element, count=None):
        Counted.__init__(self, count)
        self._element = element
        self._repr = "BasisFunction(%r, %r)" % (self._element, self._count)
    
    def element(self):
        return self._element
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._element.value_shape()
    
    def domain(self):
        return self._element.domain()
    
    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "v_%s" % count
        else:
            return "v_{%s}" % count
    
    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return isinstance(other, BasisFunction) and self._count == other._count

# --- Helper functions for pretty syntax ---

def TestFunction(element):
    return BasisFunction(element, -2)

def TrialFunction(element):
    return BasisFunction(element, -1)

# --- Helper functions for creating subfunctions on mixed elements ---

def BasisFunctions(element):
    return split(BasisFunction(element))

def TestFunctions(element):
    return split(TestFunction(element))

def TrialFunctions(element):
    return split(TrialFunction(element))
