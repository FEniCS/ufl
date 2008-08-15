"""Restriction operations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-06-08 -- 2008-08-15"

from .output import ufl_error
from .base import UFLObject


#--- Restriction operators ---

class Restricted(UFLObject):
    def __init__(self, f):
        self.f = f

    def shape(self):
        return self.f.shape()

    def operands(self):
        return (self.f,)
    
    def free_indices(self):
        return self.f.free_indices()
    
    def __str__(self):
        return "(%s)('%s')" % (self.f, self.side)

class PositiveRestricted(Restricted):
    def __init__(self, f):
        Restricted.__init__(self, f)
        self.side = "+"
    
    def __repr__(self):
        return "PositiveRestricted(%r)" % self.f

class NegativeRestricted(Restricted):
    def __init__(self, f):
        Restricted.__init__(self, f)
        self.side = "+"
    
    def __repr__(self):
        return "NegativeRestricted(%r)" % self.f

def _restrict(self, side):
    if side == "+":
        return PositiveRestricted(self)
    if side == "-":
        return NegativeRestricted(self)
    ufl_error("Invalid side %r in restriction operator." % side)
UFLObject.__call__ = _restrict

