"""This module defines the class BasisFunction and a number of related
classes (functions), including TestFunction and TrialFunction."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-23"

# Modified by Anders Logg, 2008

from ufl.assertions import ufl_assert
from ufl.common import Counted, product
from ufl.terminal import FormArgument
from ufl.split_functions import split
from ufl.finiteelement import FiniteElementBase

# --- Class representing a basis function argument in a form ---

class BasisFunction(FormArgument, Counted):
    __slots__ = ("_repr", "_element",)
    _globalcount = 0

    def __init__(self, element, count=None):
        FormArgument.__init__(self)
        Counted.__init__(self, count, BasisFunction)
        ufl_assert(isinstance(element, FiniteElementBase),
            "Expecting a FiniteElementBase instance.")
        self._element = element
        self._repr = "BasisFunction(%r, %r)" % (self._element, self._count)
    
    def reconstruct(self, count=None):
        if count is None or count == self._count:
            return self
        return BasisFunction(self.element(), count)
    
    def element(self):
        return self._element
    
    def shape(self):
        return self._element.value_shape()
    
    def cell(self):
        return self._element.cell()
    
    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "v_%s" % count
        else:
            return "v_{%s}" % count
    
    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return isinstance(other, BasisFunction) and self._element == other._element and self._count == other._count

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
