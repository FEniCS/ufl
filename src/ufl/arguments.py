
from base import *
from elements import *

### Variants of functions derived from finite elements

class BasisFunction(UFLObject):
    def __init__(self, element):
        self.element = element
    
    def __repr__(self):
        return "BasisFunction(%s)" % repr(self.element)

    def ops(self):
        return tuple()

    def fromops(self, *ops):
        return self


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

class UFLCoefficient(UFLObject):
    _count = 0
    def __init__(self, element, name):
        self.count = UFLCoefficient._count
        self.name = name
        self.element = element
        UFLCoefficient._count += 1

    def ops(self):
        return tuple()

    def fromops(self, *ops):
        return self

class Function(UFLCoefficient):
    def __init__(self, element, name):
        UFLCoefficient.__init__(self, element, name)
    
    def __repr__(self):
        return "Function(%s, %s)" % (repr(self.element), repr(self.name))

class Constant(UFLCoefficient):
    def __init__(self, polygon, name):
        UFLCoefficient.__init__(self, FiniteElement("DG", polygon, 0), name)
        self.polygon = polygon
    
    def __repr__(self):
        return "Constant(%s, %s)" % (repr(self.element.polygon), repr(self.name))

