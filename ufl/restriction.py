"""Restriction operations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-06-08 -- 2008-11-06"

from .output import ufl_error
from .base import Expr

#--- Restriction operators ---

class Restricted(Expr):
    __slots__ = ("_f", "_side")
    
    def __init__(self, f, side):
        self._f = f
        self._side = side

    def shape(self):
        return self._f.shape()

    def operands(self):
        return (self._f,)
    
    def free_indices(self):
        return self._f.free_indices()
    
    def index_dimensions(self):
        return self._f.index_dimensions()
    
    def __str__(self):
        return "(%s)('%s')" % (self._f, self._side)

class PositiveRestricted(Restricted):
    def __init__(self, f):
        Restricted.__init__(self, f, "+")
    
    def __repr__(self):
        return "PositiveRestricted(%r)" % self._f

class NegativeRestricted(Restricted):
    def __init__(self, f):
        Restricted.__init__(self, f, "-")
    
    def __repr__(self):
        return "NegativeRestricted(%r)" % self._f
