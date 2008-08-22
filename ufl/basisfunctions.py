"""Form arguments defined in finite element spaces.
There are two groups: basisfunctions and coefficients,
which use the baseclasses BasisFunction and Function."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-22"


from .output import ufl_warning, ufl_error, ufl_assert
from .base import Terminal
from .finiteelement import FiniteElement, MixedElement, VectorElement
from .common import Counted, product
from .tensors import Vector, Matrix, Tensor


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
        return "v_%d" % self._count
    
    def __repr__(self):
        return "BasisFunction(%r, %r)" % (self._element, self._count)

def TestFunction(element):
    return BasisFunction(element, -2)

def TrialFunction(element):
    return BasisFunction(element, -1)

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
    __slots__ = ("_polygon",)

    def __init__(self, polygon, name=None, count=None):
        self._polygon = polygon
        element = FiniteElement("DG", polygon, 0)
        Function.__init__(self, element, name, count)
    
    def __str__(self):
        if self._name is None:
            return "c_%d" % self._count
        else:
            return "c_%s" % self._name
    
    def __repr__(self):
        return "Constant(%r, %r, %r)" % (self._polygon, self._name, self._count)

def _foo_functions(element, function_type):
    ufl_assert(isinstance(element, MixedElement), "Expecting MixedElement instance.")
    ufl_assert(len(element.value_shape()) == 1, "MixedElement with value shape != 1 not handled!")
    
    f = function_type(element)
    value_size = product(element.value_shape())
    
    offset = 0
    subfunctions = []
    for i, e in enumerate(element.sub_elements()):
        shape = e.value_shape()
        rank = len(shape)
        size = product(shape)
        if rank == 0:
            subf = f[offset]
            offset += 1
        elif rank == 1:
            components = [f[j] for j in range(offset, offset+size)]
            subf = Vector(components)
            offset += size
        else:
            ufl_error("*Functions(element) not implemented for rank > 1 elements yet.")
            # FIXME: Need to handle symmetries etc...
        subfunctions.append(subf)
    
    ufl_assert(value_size == offset, "Logic breach in offset accumulation in *Functions.")
    return tuple(subfunctions)

def BasisFunctions(element):
    return _foo_functions(element, BasisFunction)

def TestFunctions(element):
    return _foo_functions(element, TestFunction)

def TrialFunctions(element):
    return _foo_functions(element, TrialFunction)

def Functions(element):
    return _foo_functions(element, Function)
