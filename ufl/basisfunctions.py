"""Form arguments defined in finite element spaces.
There are two groups: basisfunctions and coefficients,
which use the baseclasses BasisFunction and Function."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-20"


from base import *
from finiteelement import FiniteElement, MixedElement, VectorElement


class BasisFunction(Terminal):
    __slots__ = ("_element", "_count",)

    _globalcount = 0
    def __init__(self, element, count=None):
        self._element = element
        if count is None:
            self._count = BasisFunction._globalcount
            BasisFunction._globalcount += 1
        else:
            self._count = count
            if count >= BasisFunction._globalcount:
                BasisFunction._globalcount = count + 1
    
    def free_indices(self):
        return ()
    
    def rank(self):
        return self._element.value_rank()
    
    def __str__(self):
        return "v_%d" % self._count
    
    def __repr__(self):
        return "BasisFunction(%s, %d)" % (repr(self._element), self._count)

def TestFunction(element):
    return BasisFunction(element, -2)

def TrialFunction(element):
    return BasisFunction(element, -1)


# FIXME: Maybe we don't need these after all:
def BasisFunctions(element):
    ufl_warning("BasisFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(BasisFunction(e) for e in element.sub_elements()) # FIXME: problem with count!

def TestFunctions(element):
    ufl_warning("BasisFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(TestFunction(e) for e in element.sub_elements()) # FIXME: problem with count!

def TrialFunctions(element):
    ufl_warning("BasisFunctions isn't properly implemented.")
    if not isinstance(element, MixedElement):
        raise UFLException("Expecting MixedElement instance.")
    return tuple(TrialFunction(e) for e in element.sub_elements()) # FIXME: problem with count!


class Function(Terminal):
    __slots__ = ("_element", "_name", "_count",)

    _globalcount = 0
    def __init__(self, element, name=None, count=None):
        self._element = element
        self._name = name
        if count is None:
            self._count = Function._globalcount
            Function._globalcount += 1
        else:
            self._count = count
            if count >= Function._globalcount:
                Function._globalcount = count + 1
    
    def free_indices(self):
        return ()
    
    def rank(self):
        return self._element.value_rank()
    
    def __str__(self):
        return "w_%d" % self._count # TODO: Use name here if available.
    
    def __repr__(self):
        return "Function(%s, %s, %d)" % (repr(self._element), repr(self._name), self._count)

class Constant(Function):
    __slots__ = ("_polygon",)

    def __init__(self, polygon, name=None, count=None):
        self._polygon = polygon
        element = FiniteElement("DG", polygon, 0)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        return "c_%d" % self._count # TODO: Use name here if available.
    
    def __repr__(self):
        return "Constant(%s, %s, %d)" % (repr(self._polygon), repr(self._name), self._count)

