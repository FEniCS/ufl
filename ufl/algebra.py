"""Basic algebra operations."""

from __future__ import absolute_import
from ufl.indexing import DefaultDim

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-20 -- 2008-10-13"

# Modified by Anders Logg, 2008

from itertools import chain
from collections import defaultdict

from .output import ufl_assert, ufl_error
from .base import UFLObject, float_value, FloatValue, is_true_scalar, is_python_scalar, as_ufl
from .indexing import extract_indices, compare_shapes

#--- Algebraic operators ---

class Sum(UFLObject):
    __slots__ = ("_operands",)
    
    def __init__(self, *operands):
        ufl_assert(len(operands), "Got sum of nothing.")
        s = operands[0].shape()
        
        ufl_assert(all(compare_shapes(s, o.shape()) for o in operands),
            "Shape mismatch in sum.")
        ufl_assert(all(operands[0].free_indices() == o.free_indices() for o in operands),
            "Can't add expressions with different free indices.")
        self._operands = tuple(operands)
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._operands[0].free_indices()
    
    def shape(self):
        return self._operands[0].shape()
    
    def __str__(self):
        return "(%s)" % " + ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return "(%s)" % " + ".join(repr(o) for o in self._operands)

class Product(UFLObject):
    """The product of two or more UFL objects."""
    __slots__ = ("_operands", "_free_indices", "_repeated_indices", "_shape")
    def __init__(self, *operands):
        # Products currently defined as valid are:
        # - something multiplied with a scalar
        # - a scalar multiplied with something
        
        tmp = []
        for o in operands:
            if not (isinstance(o, FloatValue) and o._value == 1):
                tmp.append(o)
        operands = tmp
        
        ufl_assert(len(operands) >= 2, "Can't make product of nothing, should catch this before getting here.")
        self._operands = tuple(operands)
        
        # Check that we have only one non-scalar object
        # among the operands and get the shape
        shapes = 0
        shape = ()
        for o in operands:
            s = o.shape()
            if s:
                shapes += 1
                shape = s
        self._shape = shape
        ufl_assert(shapes <= 1, "Can't multiply %d non-scalar objects." % shapes)
        
        # Extract indices
        all_indices = tuple(chain(*(o.free_indices() for o in operands)))
        (self._free_indices, self._repeated_indices, dummy) = \
            extract_indices(all_indices)
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._shape
    
    def repeated_index_dimensions(self, default_dim):
        d = {}
        for i in self._repeated_indices:
            d[i] = default_dim # FIXME: Allow other dimensions here!
        return d
    
    def __str__(self):
        return "(%s)" % " * ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return "(%s)" % " * ".join(repr(o) for o in self._operands)

class Division(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        self._a = as_ufl(a)
        self._b = as_ufl(b)
        ufl_assert(is_true_scalar(self._b), "Division by non-scalar.")
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._a.free_indices()
    
    def shape(self):
        return self._a.shape()

    def is_linear(self):
        return False
    
    def __str__(self):
        return "(%s / %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%r / %r)" % (self._a, self._b)

class Power(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b),
            "Non-scalar power not defined.")
        self._a = as_ufl(a)
        self._b = as_ufl(b)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return tuple()
    
    def shape(self):
        return ()

    def is_linear(self):
        return isinstance(self._b, int)
    
    def __str__(self):
        return "(%s ** %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%r ** %r)" % (self._a, self._b)

class Abs(UFLObject):
    __slots__ = ("_a",)
    
    def __init__(self, a):
        self._a = as_ufl(a)
    
    def operands(self):
        return (self._a, )
    
    def free_indices(self):
        return self._a.free_indices()
    
    def shape(self):
        return self._a.shape()

    def is_linear(self):
        return False
    
    def __str__(self):
        return "| %s |" % str(self._a)
    
    def __repr__(self):
        return "Abs(%r)" % self._a
