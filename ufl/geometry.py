"Types for quantities computed from cell geometry."

# Copyright (C) 2008-2012 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2009
# Modified by Marie E. Rognes 2012
#
# First added:  2008-03-14
# Last changed: 2012-08-16

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
domain2num_facets = {"cell1D": None,
                     "cell2D": None,
                     "cell3D": None,
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

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return True # NB! Assuming all geometric quantities in here are are cellwise constant by default!

class SpatialCoordinate(GeometricQuantity):
    "Representation of a spatial coordinate."
    __slots__ = ("_repr",)
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)
        self._repr = "SpatialCoordinate(%r)" % self._cell

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return False

    def shape(self):
        d = self._cell.geometric_dimension()
        if d == 1:
            return ()
        return (d,)

    def evaluate(self, x, mapping, component, index_values):
        if component == ():
            if isinstance(x, (tuple,list)):
                return float(x[0])
            else:
                return float(x)
        else:
            return float(x[component[0]])

    def __str__(self):
        return "x"

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return isinstance(other, SpatialCoordinate) and other._cell == self._cell

class FacetNormal(GeometricQuantity):
    "Representation of a facet normal."
    __slots__ = ()
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)

    def shape(self):
        d = self._cell.geometric_dimension()
        return () if d == 1 else (d,)

    def __str__(self):
        return "n"

    def __repr__(self):
        return "FacetNormal(%r)" % self._cell

    def __eq__(self, other):
        return isinstance(other, FacetNormal) and other._cell == self._cell

class CellVolume(GeometricQuantity):
    "Representation of a cell volume."
    __slots__ = ()
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)

    def shape(self):
        return ()

    def __str__(self):
        return "volume"

    def __repr__(self):
        return "CellVolume(%r)" % self._cell

    def __eq__(self, other):
        return isinstance(other, CellVolume) and other._cell == self._cell

class Circumradius(GeometricQuantity):
    "Representation of the circumradius of a cell."
    __slots__ = ()
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)

    def shape(self):
        return ()

    def __str__(self):
        return "circumradius"

    def __repr__(self):
        return "Circumradius(%r)" % self._cell

    def __eq__(self, other):
        return isinstance(other, Circumradius) and other._cell == self._cell

class CellSurfaceArea(GeometricQuantity):
    "Representation of the total surface area of a cell."
    __slots__ = ()
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)

    def shape(self):
        return ()

    def __str__(self):
        return "surfacearea"

    def __repr__(self):
        return "CellSurfaceArea(%r)" % self._cell

    def __eq__(self, other):
        return isinstance(other, CellSurfaceArea) and other._cell == self._cell

class FacetArea(GeometricQuantity):
    "Representation of the area of a cell facet."
    __slots__ = ()
    def __init__(self, cell):
        GeometricQuantity.__init__(self, cell)

    def shape(self):
        return ()

    def __str__(self):
        return "facetarea"

    def __repr__(self):
        return "FacetArea(%r)" % self._cell

    def __eq__(self, other):
        return isinstance(other, FacetArea) and other._cell == self._cell

# TODO: If we include this here, we must define exactly what is meant by the mesh size, possibly adding multiple kinds of mesh sizes (hmin, hmax, havg, ?)
#class MeshSize(GeometricQuantity):
#    __slots__ = ()
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
        ufl_assert(isinstance(dimension, int), "Expecting integer.")
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
    __slots__ = ("_domain", "_space",
                 "_geometric_dimension", "_topological_dimension",
                 "_repr", "_invalid",
                 "_n", "_x", "_volume", "_circumradius",
                 "_cellsurfacearea", "_facetarea",)

    def __init__(self, domain, space=None):
        "Initialize basic cell description."

        # Check for valid domain. We allow None to support PyDOLFIN
        # integration features. This is a bit dangerous because
        # several things in UFL become undefined, but it is a very
        # good and important feature to have so don't even think about
        # removing it! ;-)
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

            # The space dimension defaults to equal
            # the topological dimension if undefined
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

        # Cache repr string
        self._repr = "Cell(%r, %r)" % (self._domain, self._space)

        # Attach expression nodes derived from this cell
        self._n = FacetNormal(self)
        self._x = SpatialCoordinate(self)
        self._volume = CellVolume(self)
        self._circumradius = Circumradius(self)
        self._cellsurfacearea = CellSurfaceArea(self)
        self._facetarea = FacetArea(self)
        #self._h = MeshSize(self)
        #self._hmin = MeshSizeMin(self)
        #self._hmax = MeshSizeMax(self)

    @property
    def x(self):
        "UFL geometry value: The global spatial coordinates."
        return self._x

    @property
    def n(self):
        "UFL geometry value: The facet normal on the cell boundary."
        return self._n

    @property
    def volume(self):
        "UFL geometry value: The volume of the cell."
        return self._volume

    @property
    def circumradius(self):
        "UFL geometry value: The circumradius of the cell."
        return self._circumradius

    @property
    def facet_area(self):
        "UFL geometry value: The area of a facet of the cell."
        return self._facetarea

    @property
    def surface_area(self):
        "UFL geometry value: The total surface area of the cell."
        return self._cellsurfacearea

    def is_undefined(self):
        """Return whether this cell is undefined,
        in which case no dimensions are available."""
        return self._invalid

    def domain(self):
        "Return the domain of the cell."
        ufl_assert(not self._invalid, "An invalid cell has no domain.")
        return self._domain

    def facet_domain(self):
        "Return the domain of the facet of this cell."
        ufl_assert(not self._invalid, "An invalid cell has no facet domains.")
        return domain2facet[self._domain]

    def num_facets(self):
        "Return the number of facets this cell has."
        ufl_assert(not self._invalid, "An invalid cell has no facets.")
        return domain2num_facets[self._domain]

    def geometric_dimension(self):
        "Return the dimension of the space this cell is embedded in."
        ufl_assert(not self._invalid, "An invalid cell has no dimensions.")
        return self._geometric_dimension

    def topological_dimension(self):
        "Return the dimension of the topology of this cell."
        ufl_assert(not self._invalid, "An invalid cell has no dimensions.")
        return self._topological_dimension

    @property
    def d(self):
        """The dimension of the cell.

        Only valid if the geometric and topological dimensions are the same."""
        ufl_assert(not self._invalid, "An invalid cell has no dimensions.")
        ufl_assert(self._topological_dimension == self._geometric_dimension,
                   "Cell.d is undefined when geometric and"+\
                   "topological dimensions are not the same.")
        return self._geometric_dimension

    def __eq__(self, other):
        return isinstance(other, Cell) and repr(self) == repr(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return "<%s cell in %s>" % (istr(self._domain), istr(self._space))

    def __repr__(self):
        return self._repr

class ProductCell(Cell):
    """Representation of a cell formed by Cartesian products of other cells."""
    __slots__ = ("_cells",)

    def __init__(self, *cells):
        "Create a ProductCell from a given list of cells."

        self._cells = cells
        ufl_assert(len(self._cells) > 0, "Expecting at least one cell")

        self._domain = self._cells[0].domain()#" x ".join([c.domain() for c in cells])
        self._invalid = False
        self._topological_dimension = sum(c.topological_dimension()
                                          for c in cells)
        self._geometric_dimension = sum(c.geometric_dimension() for c in cells)

        self._space = Space(self._topological_dimension)

        self._repr = "ProductCell(*%r)" % list(self._cells)
        self._n = None
        self._x = SpatialCoordinate(self) # For now
        self._volume = None
        self._circumradius = None
        self._cellsurfacearea = None
        self._facetarea = None            # Not defined

    def sub_cells(self):
        "Return list of cell factors."
        return self._cells

# --- Utility conversion functions

def as_cell(cell):
    """Convert any valid object to a Cell (in particular, domain string),
    or return cell if it is already a Cell."""
    return cell if isinstance(cell, Cell) else Cell(cell)
