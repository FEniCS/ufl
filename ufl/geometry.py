"Types for quantities computed from cell geometry."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-03"

# Modified by Anders Logg, 2009.

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.common import domain2dim
from ufl.terminal import Terminal

class GeometricQuantity(Terminal):
    __slots__ = ("_cell",)
    def __init__(self, cell):
        Terminal.__init__(self)
        self._cell = as_cell(cell)
    
    def cell(self):
        return self._cell

class SpatialCoordinate(GeometricQuantity):
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)

    def shape(self):
        return (self._cell.d,)
    
    def evaluate(self, x, mapping, component, index_values):
        return x[component[0]]
    
    def __str__(self):
        return "x"
    
    def __repr__(self):
        return "SpatialCoordinate(%r)" % self._cell
    
    def __eq__(self, other):
        return isinstance(other, SpatialCoordinate) and other._cell == self._cell

class FacetNormal(GeometricQuantity):
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)
    
    def shape(self):
        return (self._cell.d,)
    
    def __str__(self):
        return "n"
    
    def __repr__(self):
        return "FacetNormal(%r)" % self._cell

    def __eq__(self, other):
        return isinstance(other, FacetNormal) and other._cell == self._cell

# TODO: If we include this here, we must define exactly what is meant by the mesh size, possibly adding multiple kinds of mesh sizes (hmin, hmax, havg, ?) 
#class MeshSize(GeometricQuantity):
#    def __init__(self, cell):
#        GeometricQuantity.__init__(self, cell)
#    
#    def shape(self):
#        return ()
#    
#    def __str__(self):
#        return "h"
#    
#    def __repr__(self):
#        return "MeshSize(%r)" % self._cell
#
#    def __eq__(self, other):
#        return isinstance(other, MeshSize) and other._cell == self._cell

class Cell(object):
    "Representation of a finite element cell."
    __slots__ = ("_domain", "_degree", "d", "n", "x")
    
    def __init__(self, domain, degree=1):
        "Initialize basic cell description"
        ufl_assert(domain in domain2dim, "Invalid domain %s." % (domain,))
        if degree != 1:
            warning("High order geometries aren't implemented anywhere yet.")
        self._domain = domain
        self._degree = degree
        self.d = domain2dim[self._domain]
        self.n = FacetNormal(self)
        self.x = SpatialCoordinate(self)
        #self.h = MeshSize(self)
        #self.hmin = MeshSizeMin(self)
        #self.hmax = MeshSizeMax(self)
    
    def geometric_dimension(self):
        return self.d
    
    def topological_dimension(self):
        return self.d
    
    def domain(self):
        return self._domain
    
    def degree(self):
        return self._degree
    
    def __eq__(self, other):
        return isinstance(other, Cell) and self._domain == other._domain and self._degree == other._degree
    
    def __hash__(self):
        return hash(("Cell", self._domain, self._degree))
    
    def __str__(self):
        return "<%s of degree %d>" % (self._domain, self._degree)
    
    def __repr__(self):
        return "Cell(%r, %r)" % (self._domain, self._degree)

# --- Utility conversion functions

def as_cell(cell):
    "Convert any valid object to a Cell (in particular, domain string)."
    return cell if isinstance(cell, Cell) else Cell(cell)
