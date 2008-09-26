"""This module defines the class BasisFunction and a number of related
classes (functions), including TestFunction and TrialFunction."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-26"

# Modified by Anders Logg, 2008

from .base import Terminal
from .common import Counted, product
from .split import split

class BasisFunction(Terminal, Counted):
    __slots__ = ("_element", )
    _globalcount = 0

    def __init__(self, element, count=None):
        Counted.__init__(self, count)
        self._element = element
    
    def element(self):
        return self._element
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._element.value_shape()
    
    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "v_%s" % count
        else:
            return "v_{%s}" % count
    
    def __repr__(self):
        return "BasisFunction(%r, %r)" % (self._element, self._count)

def TestFunction(element):
    return BasisFunction(element, -2)

def TrialFunction(element):
    return BasisFunction(element, -1)

def BasisFunctions(element):
    return split(BasisFunction(element))

def TestFunctions(element):
    return split(TestFunction(element))

def TrialFunctions(element):
    return split(TrialFunction(element))
