
from base import *

### Quantities computed from cell geometry

class GeometricQuantity(UFLObject):
    def __init__(self):
        pass

    def ops(self):
        return tuple()
    
    def fromops(self, ops):
        return self
    

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


