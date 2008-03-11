#!/usr/bin/env python

"""
Types for quantities computed from cell geometry.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 11th 2008"

from base import *


class GeometricQuantity(Terminal):
    def __init__(self):
        pass
    

class FacetNormal(GeometricQuantity):
    def __init__(self):
        pass
        self.free_indices = tuple()
        self.rank = 1

    def __repr__(self):
        return "FacetNormal()"

class MeshSize(GeometricQuantity):
    def __init__(self):
        pass
        self.free_indices = tuple()
        self.rank = 0

    def __repr__(self):
        return "MeshSize()"

# TODO: More mesh quantities? Local measures of mesh quality should be easy to add, if we just define what's interesting.

