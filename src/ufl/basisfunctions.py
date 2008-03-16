#!/usr/bin/env python

"""
Form arguments defined in finite element spaces.
There are two groups: basisfunctions and coefficients,
which use the baseclasses BasisFunction and Coefficient.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-14-03 -- 2008-16-03"

from base import *
from elements import *

class BasisFunction(Terminal):
    __slots__ = ("element", "name", "count",)

    _globalcount = 0
    def __init__(self, element, count=None):
        self.element = element
        if count is None:
            self.count = BasisFunction._globalcount
            BasisFunction._globalcount += 1
        else:
            self.count = count
            if count >= BasisFunction._globalcount:
                BasisFunction._globalcount = count + 1
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return self.element.value_rank()
    
    def __str__(self):
        return "BasisFunction(%s)" % str(self.element)
    
    def __repr__(self):
        return "BasisFunction(%s, %d)" % (repr(self.element), self.count)

def TestFunction(element):
    return BasisFunction(element, -2)

def TrialFunction(element):
    return BasisFunction(element, -1)


# FIXME: Maybe we don't need these afterall:
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
    __slots__ = ("element", "name", "count",)

    _globalcount = 0
    def __init__(self, element, name=None, count=None):
        self.element = element
        self.name = name
        if count is None:
            self.count = Function._globalcount
            Function._globalcount += 1
        else:
            self.count = count
            if count >= Function._globalcount:
                Function._globalcount = count + 1
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return self.element.value_rank()
    
    def __str__(self):
        return "Function(%s)" % str(self.element) # TODO: Better pretty print. Use name here?
    
    def __repr__(self):
        return "Function(%s, %s, %d)" % (repr(self.element), repr(self.name), self.count)

class Constant(Function):
    __slots__ = ("polygon",)

    def __init__(self, polygon, name=None, count=None):
        self.polygon = polygon
        element = FiniteElement("DG", polygon, 0)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        return "Constant(%d)" % self.count # TODO: Better pretty print. Use name here?
    
    def __repr__(self):
        return "Constant(%s, %s, %d)" % (repr(self.polygon), repr(self.name), self.count)

