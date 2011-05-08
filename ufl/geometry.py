"Types for quantities computed from cell geometry."

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-03-14 -- 2011-04-28"

# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2009

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.common import istr
from ufl.terminal import Terminal

# --- Expression node types

# Mapping from domain (cell) to dimension
domain2dim = {"cell1D": 1,
              "cell2D": 2,
              "cell3D": 3,
              "vertex": 0,
              "interval": 1,
              "triangle": 2,
              "tetrahedron": 3,
              "quadrilateral": 2,
              "hexahedron": 3}

# Mapping from domain (cell) to facet
domain2facet = {"cell1D": "vertex",
                "cell2D": "cell1D",
                "cell3D": "cell2D",
                "interval": "vertex",
                "triangle": "interval",
                "tetrahedron": "triangle",
                "quadrilateral": "interval",
                "hexahedron": "quadrilateral"}

# Number of facets associated with each domain
domain2num_facets = {"interval": 2,
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
        d = self._cell.geometric_dimension()
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
        d = self._cell.geometric_dimension()
        if d == 1:
            return ()
        return (d,)

    def __str__(self):
        return "n"

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return isinstance(other, FacetNormal) and other._cell == self._cell

class CellVolume(GeometricQuantity):
    "Representation of a cell volume."
    __slots__ = ("_repr",)
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)
        self._repr = "CellVolume(%r)" % self._cell

    def shape(self):
        return ()

    def __str__(self):
        return "volume"

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return isinstance(other, CellVolume) and other._cell == self._cell

class Circumradius(GeometricQuantity):
    "Representation of the circumradius of a cell."
    __slots__ = ("_repr",)
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)
        self._repr = "Circumradius(%r)" % self._cell

    def shape(self):
        return ()

    def __str__(self):
        return "circumradius"

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return isinstance(other, Circumradius) and other._cell == self._cell

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
        #ufl_assert(isinstance(dimension, int), "Expecting integer.") # FIXME: This is essential!
        self._dimension = dimension

    def dimension(self):
        ufl_assert(isinstance(self._dimension, int), "No dimension defined!")
        return self._dimension

    def __str__(self):
        return "R%s" % istr(self._dimension)

    def __repr__(self):
        return "Space(%r)" % (self._dimension,)

class Cell(object):
    "Representation of a finite element cell."
    __slots__ = ("_domain", "_degree", "_space", "_geometric_dimension",
                 "_topological_dimension", "_repr", "_invalid",
                 "d", "n", "x", "volume", "circumradius")

    def __init__(self, domain, degree=1, space=None):
        "Initialize basic cell description"

        # Check for valid domain, for now we allow None to support
        # PyDOLFIN integration features, but this is a bit dangerous
        # because several things in UFL become undefined...
        if domain is None:
            self._invalid = True
        else:
            ufl_assert(domain in domain2dim, "Invalid domain %s." % (domain,))
            self._invalid = False
        self._domain = domain

        # Don't compute quantities that are undefined
        if self._invalid:
            # Used in repr string below
            self._space = None
        else:
            # The topological dimension is defined by the cell type
            dim = domain2dim[self._domain]
            self._topological_dimension = dim

            # The space dimension defaults to equal the topological dimension if undefined
            space = Space(dim) if space is None else space
            ufl_assert(isinstance(space, Space),
                       "Expecting a Space instance, not '%r'" % (space,))
            self._space = space

            # The geometric dimension is defined by the space
            self._geometric_dimension = space.dimension()

            # Check for consistency in dimensions.
            # NB! Note that the distinction between topological
            # and geometric dimensions has yet to be used in
            # practice, so don't trust it too much :)
            ufl_assert(self._topological_dimension <= self._geometric_dimension,
                       "Cannot embed a %sD cell in %s" %\
                           (istr(self._topological_dimension), self._space))

            # Attach a cell dimension for use in code.
            # TODO: Make self.d a property, deprecate or make valid
            #       only in this case. Don't use inside UFL!
            if self._topological_dimension == self._geometric_dimension:
                self.d = self._geometric_dimension
            else:
                self.d = None # TODO: Make this a property to fail instead of silently getting None

        # Handle degree TODO: Remove degree from cell completely
        ufl_assert(isinstance(degree, int) and degree >= 1, "Invalid degree '%r'." % (degree,))
        if degree != 1: # TODO: Remove warning when implemented
            warning("Note: High order geometries are not implemented in the form compilers yet.")
        self._degree = degree

        # Cache repr string
        self._repr = "Cell(%r, %r, %r)" % (self._domain, self._degree, self._space)

        # Attach expression nodes derived from this cell
        self.n = FacetNormal(self)
        self.x = SpatialCoordinate(self)
        self.volume = CellVolume(self)
        self.circumradius = Circumradius(self)
        #self.h = MeshSize(self)
        #self.hmin = MeshSizeMin(self)
        #self.hmax = MeshSizeMax(self)

    def geometric_dimension(self):
        ufl_assert(not self._invalid, "An invalid cell has no dimensions.")
        return self.d

    def topological_dimension(self):
        ufl_assert(not self._invalid, "An invalid cell has no dimensions.")
        return self.d

    def is_undefined(self):
        return self._invalid

    def domain(self):
        ufl_assert(not self._invalid, "An invalid cell has no domain.")
        return self._domain

    def degree(self):
        ufl_assert(not self._invalid, "An invalid cell has no degree.")
        return self._degree

    def num_facets(self):
        ufl_assert(not self._invalid, "An invalid cell has no facets.")
        return domain2num_facets[self._domain]

    def facet_domain(self):
        ufl_assert(not self._invalid, "An invalid cell has no facet domains.")
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
