"Types for quantities computed from cell geometry."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14"

# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2009
# Last changed: 2009-12-21

from ufl.log import warning
from ufl.assertions import ufl_assert
#from ufl.common import domain2dim, istr
from ufl.common import istr
from ufl.terminal import Terminal

# --- Expression node types

# Mapping from domain (cell) to dimension
domain2dim = {None: None,
              "vertex": 0,
              "interval": 1,
              "triangle": 2,
              "tetrahedron": 3,
              "quadrilateral": 2,
              "hexahedron": 3,
              "facet": 0}

# Mapping from domain (cell) to facet
domain2facet = {None: None,
                "interval": "vertex",
                "triangle": "interval",
                "tetrahedron": "triangle",
                "quadrilateral": "interval",
                "hexahedron": "quadrilateral"}

# Number of facets associated with each domain
domain2num_facets = {None: None,
                     "interval": 2,
                     "triangle": 3,
                     "tetrahedron": 4,
                     "quadrilateral": 4,
                     "hexahedron": 6}

# Valid UFL domains
ufl_domains = tuple(domain2dim.keys())

class GeometricQuantity(Terminal):
    __slots__ = ("_cell",)
    def __init__(self, cell):
        Terminal.__init__(self)
        self._cell = as_cell(cell)

    def cell(self):
        return self._cell

class SpatialCoordinate(GeometricQuantity):
    "Representation of a spatial coordinate."
    __slots__ = ("_repr",)
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)
        self._repr = "SpatialCoordinate(%r)" % self._cell

    def shape(self):
        d = self._cell.d
        if d == 1:
            return ()
        return (d,)

    def evaluate(self, x, mapping, component, index_values):
        return float(x[component[0]])

    def __str__(self):
        return "x"

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return isinstance(other, SpatialCoordinate) and other._cell == self._cell

class FacetNormal(GeometricQuantity):
    "Representation of a facet normal."
    __slots__ = ("_repr",)
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)
        self._repr = "FacetNormal(%r)" % self._cell

    def shape(self):
        d = self._cell.d
        if d == 1:
            return ()
        return (d,)

    def __str__(self):
        return "n"

    def __repr__(self):
        return self._repr

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

# --- Basic space and cell representation classes

class Space(object):
    "Representation of an Euclidean space."
    __slots__ = ("_dimension",)

    def __init__(self, dimension):
        #ufl_assert(isinstance(dimension, int), "Expecting integer.")
        self._dimension = dimension

    def dimension(self):
        return self._dimension

    def __str__(self):
        return "R%s" % istr(self._dimension)

    def __repr__(self):
        return "Space(%s)" % istr(self._dimension)

class Cell(object):
    "Representation of a finite element cell."
    __slots__ = ("_domain", "_degree", "_space", "_geometric_dimension",
                 "_topological_dimension", "_repr", "d", "n", "x")

    def __init__(self, domain, degree=1, space=None):
        "Initialize basic cell description"

        # Handle domain
        ufl_assert(domain in domain2dim, "Invalid domain %s." % (domain,))
        self._domain = domain
        self._topological_dimension = domain2dim[self._domain]

        # Handle degree
        ufl_assert(isinstance(degree, int) and degree >= 1, "Invalid degree '%r'." % (degree,))
        if degree != 1: # TODO: Remove warning when implemented
            warning("Note: High order geometries are not implemented in the form compilers yet.")
        self._degree = degree

        # Get geometric dimension
        if space is None:
            space = Space(self._topological_dimension)
        ufl_assert(isinstance(space, Space), "Expecting a Space instance, not '%r'" % (space,))
        self._space = space
        self._geometric_dimension = self._space.dimension()

        ufl_assert(self._topological_dimension <= self._geometric_dimension,
            "Cannot embed a %sD cell in %s" % (istr(self._topological_dimension), self._space))

        # TODO: Make self.d a property, deprecate or make valid
        #       only in this case. Don't use inside UFL!
        if self._topological_dimension == self._geometric_dimension:
            self.d = self._geometric_dimension
        else:
            self.d = None

        # Cache repr string
        self._repr = "Cell(%r, %r, %r)" % (self._domain, self._degree, self._space)

        # Attach expression nodes derived from this cell
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

    def num_facets(self):
        return domain2num_facets[self._domain]

    def facet_domain(self):
        return domain2facet[self._domain]

    def __eq__(self, other):
        return isinstance(other, Cell) and self._domain == other._domain and self._degree == other._degree

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(("Cell", self._domain, self._degree))

    def __str__(self):
        return "<%s of degree %d>" % (istr(self._domain), self._degree)

    def __repr__(self):
        return self._repr

# --- Utility conversion functions

def as_cell(cell):
    "Convert any valid object to a Cell (in particular, domain string)."
    return cell if isinstance(cell, Cell) else Cell(cell)
