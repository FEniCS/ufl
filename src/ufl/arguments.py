#!/usr/bin/env python

"""
Form arguments defined in finite element spaces.
There are two groups: basisfunctions and coefficients,
which use the baseclasses BasisFunction and Coefficient.
"""

from base import *
from elements import *


class BasisFunction(Terminal):
    def __init__(self, element):
        self.element = element
    
    def __repr__(self):
        return "BasisFunction(%s)" % repr(self.element)


def BasisFunctions(element):
    if isinstance(element, MixedElement):
        return tuple(BasisFunction(fe) for fe in element.elements)
    raise ValueError("Expecting MixedElement instance.")

class TestFunction(BasisFunction):
    def __init__(self, element):
        self.element = element

    def __repr__(self):
        return "TestFunction(%s)" % repr(self.element)

def TestFunctions(element):
    if isinstance(element, MixedElement):
        return tuple(TestFunction(fe) for fe in element.elements)
    raise ValueError("Expecting MixedElement instance.")

class TrialFunction(BasisFunction):
    def __init__(self, element):
        self.element = element

    def __repr__(self):
        return "TrialFunction(%s)" % repr(self.element)

def TrialFunctions(element):
    if isinstance(element, MixedElement):
        return tuple(TrialFunction(fe) for fe in element.elements)
    raise ValueError("Expecting MixedElement instance.")

class Coefficient(Terminal):
    _count = 0
    def __init__(self, element, name):
        self.count = Coefficient._count
        self.name = name
        self.element = element
        Coefficient._count += 1

class Function(Coefficient):
    def __init__(self, element, name):
        Coefficient.__init__(self, element, name)
    
    def __repr__(self):
        return "Function(%s, %s)" % (repr(self.element), repr(self.name))

class Constant(Coefficient):
    def __init__(self, polygon, name):
        Coefficient.__init__(self, FiniteElement("DG", polygon, 0), name)
        self.polygon = polygon
    
    def __repr__(self):
        return "Constant(%s, %s)" % (repr(self.element.polygon), repr(self.name))

