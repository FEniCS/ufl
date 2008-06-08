"""Restriction operations."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-06-08 -- 2008-06-08"

from output import ufl_error
from base import UFLObject


#--- Restriction operators ---

class Restricted(UFLObject):
    def __init__(self, f):
        self.f = f

    def rank(self):
        return self.f.rank()

    def operands(self):
        return (self.f,)
    
    def free_indices(self):
        return self.f.free_indices()
    
    def __str__(self):
        return "(%s)('%s')" % (str(self.f), self.side)

class PositiveRestricted(Restricted):
    def __init__(self, f):
        Restricted.__init__(self, f)
        self.side = "+"
    
    def __repr__(self):
        return "PositiveRestricted(%s)" % repr(self.f)

class NegativeRestricted(Restricted):
    def __init__(self, f):
        Restricted.__init__(self, f)
        self.side = "+"
    
    def __repr__(self):
        return "NegativeRestricted(%s)" % repr(self.f)

def _restrict(self, side):
    if side == "+":
        return PositiveRestricted(self)
    if side == "-":
        return NegativeRestricted(self)
    ufl_error("Invalid side %s in restriction operator." % repr(side))
UFLObject.__call__ = _restrict

