#!/usr/bin/env python

"""
Types for quantities computed from cell geometry.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 8th 2008"

from base import *


class GeometricQuantity(Terminal):
    def __init__(self):
        pass
    

class FacetNormal(GeometricQuantity):
    def __init__(self):
        pass
        #self.free_indices = (Index(...),) # FIXME

    def __repr__(self):
        return "FacetNormal()"

class CellRadius(GeometricQuantity):
    def __init__(self):
        pass
        #self.free_indices = (Index(...),) # FIXME

    def __repr__(self):
        return "CellRadius()"


