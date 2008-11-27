"This module defines the Zero class."


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-11-01 -- 2008-11-07"

from ufl.output import ufl_assert
from ufl.base import Terminal
from ufl.indexing import Index

#--- Class for representing zero tensors of different shapes ---

_zero_cache = {}
class Zero(Terminal):
    __slots__ = ("_shape", "_free_indices", "_index_dimensions")
    
    def __new__(cls, shape=(), free_indices=(), index_dimensions=None):
        # check cache to reuse objects
        if index_dimensions is None:
            index_dimensions = {}
        else:
            index_dimensions = dict(index_dimensions)
        ufl_assert(all(isinstance(i, Index) for i in free_indices),
                   "Expecting tuple if Index objects.")
        ufl_assert(not(set(free_indices) ^ set(index_dimensions.keys())),
                   "Index set mismatch.")
        key = (shape, free_indices, tuple(index_dimensions.items()))
        z = _zero_cache.get(key, None)
        if z is not None:
            return z
        # construct new instance
        self = Terminal.__new__(cls)
        self._init(shape, free_indices, index_dimensions)
        _zero_cache[key] = self
        return self
    
    def _init(self, shape, free_indices, index_dimensions):
        self._shape = shape
        self._free_indices = free_indices
        self._index_dimensions = index_dimensions
    
    def __init__(self, shape=(), free_indices=(), index_dimensions=None):
        Terminal.__init__(self)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._shape
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def __str__(self):
        return "[Zero tensor with shape %s and free indices %s]" % \
            (repr(self._shape), repr(self._free_indices))
    
    def __repr__(self):
        return "Zero(%s, %s, %s)" % (repr(self._shape),
            repr(self._free_indices), repr(self._index_dimensions))
    
    def __eq__(self, other):
        # zero is zero no matter which free indices you look at
        if self._shape == () and other == 0:
            return True
        return isinstance(other, Zero) and self._shape == other.shape()
    
    def __neg__(self):
        return self
    
    def __abs__(self):
        return self
    
    def __nonzero__(self):
        return False 

