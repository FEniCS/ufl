"This module defines the Zero class."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-11-01 -- 2008-11-01"

from .base import Terminal

#--- Class for representing zero tensors of different shapes ---

_zero_cache = {}
class Zero(Terminal):
    __slots__ = ("_shape",)
    
    def __new__(cls, shape=()):
        global _zero_cache
        # check cache to reuse objects
        z = _zero_cache.get(shape, None)
        if z is not None: return z
        # construct new instance
        self = Terminal.__new__(cls)
        self._init(shape)
        _zero_cache[shape] = self
        return self
    
    def _init(self, shape):
        self._shape = shape
    
    def __init__(self, shape=()):
        pass
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        return "[Zero tensor with shape %s]" % repr(self._shape)
    
    def __repr__(self):
        return "Zero(%s)" % repr(self._shape)
    
    def __eq__(self, other):
        if self._shape == () and other == 0:
            return True
        return isinstance(other, Zero) and self._shape == other._shape
    
    def __neg__(self):
        return self
    
    def __abs__(self):
        return self
    
    def __nonzero__(self):
        return False 

