#!/usr/bin/env python

"""
Form arguments defined in finite element spaces.
There are two groups: basisfunctions and coefficients,
which use the baseclasses BasisFunction and Coefficient.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 8th 2008"

from base import *
from elements import *


class BasisFunction(Terminal):
    count = 0
    def __init__(self, element, count=None):
        self.element = element
        if count is None:
            self.count = BasisFunction.count
            BasisFunction.count += 1
        else:
            self.count = count
        
        self.free_indices = tuple(Index() for i in range(self.element.value_rank))
    
    def __repr__(self):
        return "BasisFunction(%s, %d)" % (repr(self.element), self.count)

def TestFunction(element):
    return BasisFunction(element, -2)

def TrialFunction(element):
    return BasisFunction(element, -1)


def BasisFunctions(element):
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(BasisFunction(e) for e in element.elements) # FIXME: problem with count!

def TestFunctions(element):
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(TestFunction(e) for e in element.elements) # FIXME: problem with count!

def TrialFunctions(element):
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(TrialFunction(e) for e in element.elements) # FIXME: problem with count!


class Function(Terminal):
    count = 0
    def __init__(self, element, name=None, count=None):
        self.element = element
        self.name = name
        if count is None:
            self.count = Function.count
            Function.count += 1
        else:
            self.count = count
        self.free_indices = tuple(Index() for i in range(self.element.value_rank))
    
    def __repr__(self):
        return "Function(%s, %s, %d)" % (repr(self.element), repr(self.name), self.count)

class Constant(Function):
    def __init__(self, polygon, name=None, count=None):
        self.polygon = polygon
        element = FiniteElement("DG", polygon, 0)
        Function.__init__(self, element, name, count)
    
    def __repr__(self):
        return "Constant(%s, %s, %d)" % (repr(self.polygon), repr(self.name), self.count)

