"Types for quantities computed from cell geometry."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-12-18"

from ufl.output import ufl_assert
from ufl.common import domain2dim
from ufl.base import Terminal

class SpatialCoordinate(Terminal):
    __slots__ = ("_domain",)
    def __init__(self, domain):
        self._domain = domain

    def shape(self):
        return (domain2dim[self._domain],)
    
    def domain(self):
        return self._domain

    def __str__(self):
        return "x"
    
    def __repr__(self):
        return "SpatialCoordinate(%r)" % self._domain

    def __eq__(self, other):
        return isinstance(other, SpatialCoordinate) and other._domain == self._domain

class FacetNormal(Terminal):
    def __init__(self, domain):
        Terminal.__init__(self)
        self._domain = domain
    
    def shape(self):
        return (domain2dim[self._domain],)
    
    def domain(self):
        return self._domain
    
    def __str__(self):
        return "n"
    
    def __repr__(self):
        return "FacetNormal(%r)" % self._domain

    def __eq__(self, other):
        return isinstance(other, FacetNormal) and other._domain == self._domain

# TODO: Do we want this? For higher degree geometry. Is this general enough?
class Cell(object):
    "Representation of a finite element cell."
    __slots__ = ("_domain", "_degree")
    
    def __init__(self, domain, degree=1):
        "Initialize basic cell description"
        ufl_assert(domain in domain2dim, "Invalid domain %s." % (domain,))
        self._domain = domain
        self._degree = degree
    
    def domain(self):
        return self._domain
    
    def degree(self):
        return self._degree
    
    def dim(self):
        return domain2dim[self._domain]
    
    def n(self):
        return FacetNormal(self._domain)
    
    def x(self):
        return SpatialCoordinate(self._domain)
    
    def __str__(self):
        return "[%s of degree %d]" % (self._domain, self._degree)
    
    def __repr__(self):
        return "Cell(%r, %r)" % (self._domain, self._degree)

# Predefined linear cells
interval      = Cell("interval")
triangle      = Cell("triangle")
tetrahedron   = Cell("tetrahedron")
quadrilateral = Cell("quadrilateral")
hexahedron    = Cell("hexahedron")

def as_cell(cell):
    "Convert any valid object to a Cell (in particular, domain string)."
    return cell if isinstance(cell, Cell) else Cell(cell)
