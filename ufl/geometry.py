"Types for quantities computed from cell geometry."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-01-09"

from ufl.output import ufl_assert, ufl_warning
from ufl.common import domain2dim
from ufl.terminal import Terminal

class SpatialCoordinate(Terminal):
    __slots__ = ("_cell",)
    def __init__(self, cell):
        self._cell = as_cell(cell)

    def shape(self):
        return (self._cell.dim(),)
    
    def cell(self):
        return self._cell
    
    def evaluate(self, x, mapping, component, index_values):
        return x[component[0]]
    
    def __str__(self):
        return "x"
    
    def __repr__(self):
        return "SpatialCoordinate(%r)" % self._cell
    
    def __eq__(self, other):
        return isinstance(other, SpatialCoordinate) and other._cell == self._cell

class FacetNormal(Terminal):
    def __init__(self, cell):
        Terminal.__init__(self)
        self._cell = as_cell(cell)
    
    def shape(self):
        return (self._cell.dim(),)
    
    def cell(self):
        return self._cell
    
    def __str__(self):
        return "n"
    
    def __repr__(self):
        return "FacetNormal(%r)" % self._cell

    def __eq__(self, other):
        return isinstance(other, FacetNormal) and other._cell == self._cell

class Cell(object):
    "Representation of a finite element cell."
    __slots__ = ("_domain", "_degree", "n", "x")
    
    def __init__(self, domain, degree=1):
        "Initialize basic cell description"
        ufl_assert(domain in domain2dim, "Invalid domain %s." % (domain,))
        if degree != 1:
            ufl_warning("High order geometries aren't implemented anywhere yet.")
        self._domain = domain
        self._degree = degree
        self.n = FacetNormal(self)
        self.x = SpatialCoordinate(self)
    
    def domain(self):
        return self._domain
    
    def degree(self):
        return self._degree
    
    # TODO: Swap this with geometric_dimension and topological_dimension
    def dim(self):
        return domain2dim[self._domain]
    
    def __eq__(self, other):
        return isinstance(other, Cell) and self._domain == other._domain and self._degree == other._degree
    
    def __hash__(self):
        return hash(("Cell", self._domain, self._degree))
    
    def __str__(self):
        return "[%s of degree %d]" % (self._domain, self._degree)
    
    def __repr__(self):
        return "Cell(%r, %r)" % (self._domain, self._degree)

# --- Utility conversion functions

def as_cell(cell):
    "Convert any valid object to a Cell (in particular, domain string)."
    return cell if isinstance(cell, Cell) else Cell(cell)

