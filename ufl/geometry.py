"""Types for quantities computed from cell geometry."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-27"

from .base import Terminal
from .indexing import DefaultDim

class FacetNormal(Terminal):
    def __init__(self):
        pass
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return (DefaultDim,)
    
    def __str__(self):
        return "n"
    
    def __repr__(self):
        return "FacetNormal()"

    def __eq__(self, other):
        return isinstance(other, FacetNormal)
