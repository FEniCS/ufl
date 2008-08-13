"""Types for quantities computed from cell geometry."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-13"

from .base import Terminal

class FacetNormal(Terminal):
    def __init__(self):
        pass
    
    def free_indices(self):
        return ()
    
    def rank(self):
        return 1
    
    def __str__(self):
        return "n"
    
    def __repr__(self):
        return "FacetNormal()"

