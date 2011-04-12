"""Restriction operations."""

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-06-08 -- 2009-02-13"

from ufl.log import error
from ufl.expr import Operator
from ufl.precedence import parstr

#--- Restriction operators ---

class Restricted(Operator):
    __slots__ = ("_f", "_side")
    
    def __init__(self, f, side):
        Operator.__init__(self)
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
    
    def evaluate(self, x, mapping, component, index_values):    
        a = self._f.evaluate(x, mapping, component, index_values)
        return a
    
    def __str__(self):
        return "%s('%s')" % (parstr(self._f, self), self._side)

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
