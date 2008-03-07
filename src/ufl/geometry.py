#!/usr/bin/env python

"""
Types for quantities computed from cell geometry.
"""

from base import *


class GeometricQuantity(Terminal):
    def __init__(self):
        pass
    

class FacetNormal(GeometricQuantity):
    def __init__(self):
        pass

    def __repr__(self):
        return "FacetNormal()"

class CellRadius(GeometricQuantity):
    def __init__(self):
        pass

    def __repr__(self):
        return "CellRadius()"


