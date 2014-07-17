"Types for representing a cell."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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
# Modified by Andrew T. T. McRae, 2014

from itertools import chain
from collections import defaultdict

from ufl.log import warning, error, deprecate
from ufl.assertions import ufl_assert
from ufl.common import istr, EmptyDict
from ufl.terminal import Terminal
from ufl.protocols import id_or_none

# --- Basic cell properties

# Mapping from cell name to topological dimension
cellname2dim = {
    "cell0D": 0,
    "cell1D": 1,
    "cell2D": 2,
    "cell3D": 3,
    "vertex":        0,
    "interval":      1,
    "triangle":      2,
    "tetrahedron":   3,
    "quadrilateral": 2,
    "hexahedron":    3,
    }

def cell2dim(cell):
    "Maps from UFL cell or cell name to topological dimension"
    if isinstance(cell, str):
        # Backwards compatibility
        cellname = cell
    else:
        cellname = cell.cellname()

    if cellname == "OuterProductCell":
        return cell2dim(cell._A) + cell2dim(cell._B)
    else:
        return cellname2dim[cellname]

# Mapping from cell name to facet name
cellname2facetname = {
    "cell0D": None,
    "cell1D": "vertex",
    "cell2D": "cell1D",
    "cell3D": "cell2D",
    "vertex":        None,
    "interval":      "vertex",
    "triangle":      "interval",
    "tetrahedron":   "triangle",
    "quadrilateral": "interval",
    "hexahedron":    "quadrilateral",
    "OuterProductCell": None
    }

reference_cell_volume = {
    "vertex": 0.0,
    "interval": 1.0,
    "triangle": 0.5,
    "tetrahedron": 1.0/6.0,
    "quadrilateral": 1.0,
    "hexahedron": 1.0
    }

affine_cells = {"vertex", "interval", "triangle", "tetrahedron"}

# Valid UFL cellnames
ufl_cellnames = tuple(sorted(chain(cellname2dim.keys(), ["OuterProductCell"])))


# --- Basic cell representation classes

class Cell(object):
    "Representation of a finite element cell."
    __slots__ = ("_cellname",
                 "_geometric_dimension",
                 "_topological_dimension"
                 )
    def __init__(self, cellname, geometric_dimension=None, topological_dimension=None):
        "Initialize basic cell description."

        # The topological dimension is defined by the cell type,
        # so the cellname must be among the known ones,
        # so we can find the known dimension, unless we have
        # a product cell, in which the given dimension is used
        tdim = cellname2dim.get(cellname, topological_dimension)

        # The geometric dimension defaults to equal the topological
        # dimension if undefined
        if geometric_dimension is None:
            gdim = tdim
        else:
            gdim = geometric_dimension

        # Validate dimensions
        ufl_assert(isinstance(gdim, int),
                   "Expecting integer dimension, not '%r'" % (gdim,))
        ufl_assert(isinstance(tdim, int),
                   "Expecting integer dimension, not '%r'" % (tdim,))
        ufl_assert(tdim <= gdim,
                   "Topological dimension cannot be larger than geometric dimension.")

        # ... Finally store validated data
        self._cellname = cellname
        self._topological_dimension = tdim
        self._geometric_dimension = gdim

    def geometric_dimension(self):
        "Return the dimension of the space this cell is embedded in."
        return self._geometric_dimension

    def topological_dimension(self):
        "Return the dimension of the topology of this cell."
        return self._topological_dimension

    def cellname(self):
        "Return the cellname of the cell."
        return self._cellname

    def facet_cellname(self):
        "Return the cellname of the facet of this cell."
        facet_cellname = cellname2facetname.get(self._cellname)
        ufl_assert(facet_cellname is not None,
                   "Name of facet cell not available for cell type %s." % self._cellname)
        return facet_cellname

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        s = (self._geometric_dimension, self._topological_dimension, self._cellname)
        o = (other._geometric_dimension, other._topological_dimension, other._cellname)
        return s == o

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, Cell):
            return False
        s = (self._geometric_dimension, self._topological_dimension, self._cellname)
        o = (other._geometric_dimension, other._topological_dimension, other._cellname)
        return s < o

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return "<%s cell in %sD>" % (istr(self._cellname),
                                     istr(self._geometric_dimension))

    def __repr__(self):
        return "Cell(%r, %r)" % (self._cellname, self._geometric_dimension)

    def _repr_svg_(self):
        ""

        name = self.cellname()
        m = 200
        if name == "interval":
            points = [(0, 0), (m, 0)]
        elif name == "triangle":
            points = [(0, m), (m, m), (0, 0), (0, m)]
        elif name == "quadrilateral":
            points = [(0, m), (m, m), (m, 0), (0, 0), (0, m)]
        else:
            points = None

        svg = '''
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%s" height="%s">
        <polyline points="%s" style="%s" />
        </svg>
        '''

        if points:
            fill = "none"
            stroke = "black"
            strokewidth = 3

            width = max(p[0] for p in points) - min(p[0] for p in points)
            height = max(p[1] for p in points) - min(p[1] for p in points)
            width = max(width, strokewidth)
            height = max(height, strokewidth)
            style = "fill:%s; stroke:%s; stroke-width:%s" % (fill, stroke, strokewidth)
            points = " ".join(','.join(map(str, p)) for p in points)
            return svg % (width, height, points, style)
        else:
            return None

class ProductCell(Cell):
    __slots__ = ("_cells",)
    def __init__(self, *cells):
        cells = tuple(as_cell(cell) for cell in cells)
        gdim = sum(cell.geometric_dimension() for cell in cells)
        tdim = sum(cell.topological_dimension() for cell in cells)
        Cell.__init__(self, "product", gdim, tdim)
        self._cells = tuple(cells)

    def sub_cells(self):
        "Return list of cell factors."
        return self._cells

    def facet_cellname(self):
        "Return the cellname of the facet of this cell."
        error("Makes no sense for product cell.")

    def __eq__(self, other):
        if not isinstance(other, ProductCell):
            return False
        return self._cells == other._cells

    def __lt__(self, other):
        if not isinstance(other, ProductCell):
            return False
        return self._cells < other._cells

    def __repr__(self):
        return "ProductCell(*%r)" % (self._cells,)


class OuterProductCell(Cell):
    """Representation of a cell formed as the Cartesian product of
    two existing cells"""
    __slots__ = ("_A", "_B", "facet_horiz", "facet_vert")

    def __init__(self, A, B, gdim=None):
        self._A = A
        self._B = B

        tdim = A.topological_dimension() + B.topological_dimension()
        # default gdim -- "only as big as it needs to be, but not smaller than A or B"
        gdim_temp = max(A.geometric_dimension(),
                        B.geometric_dimension(),
                        A.topological_dimension() + B.topological_dimension())
        if gdim is None:
            # default gdim
            gdim = gdim_temp
        else:
            # otherwise, validate custom gdim
            if not isinstance(gdim, int):
                raise TypeError("gdim must be an integer")
            if gdim < gdim_temp:
                raise ValueError("gdim must be at least %d" % gdim_temp)
        Cell.__init__(self, "OuterProductCell", gdim, tdim)

        # facets for extruded cells
        if B.cellname() == "interval":
            self.facet_horiz = A
            if A.topological_dimension() == 2:
                self.facet_vert = OuterProductCell(Cell("interval"), Cell("interval"))
            elif A.topological_dimension() == 1:
                # Terminate this recursion somewhere!
                self.facet_vert = Cell("interval")
            else:
                # Don't know how to extrude this
                self.facet_vert = None

    def facet_cellname(self):
        "Return the cellname of the facet of this cell."
        error("Makes no sense for OuterProductCell.")

    def __eq__(self, other):
        if not isinstance(other, OuterProductCell):
            return False
        # This is quite subtle: my intuition says that the OPCs of
        # Cell("triangle") with Cell("interval"), and
        # Cell("triangle", 3) with Cell("interval")
        # are essentially the same: triangular prisms with gdim = tdim = 3.
        # For safety, though, we will only compare equal if the
        # subcells are *identical*, including immersion.
        return self._A == other._A and self._B == other._B and self.geometric_dimension() == other.geometric_dimension()

    def __lt__(self, other):
        # No idea what this might be used for
        error("Makes no sense for OuterProductCell")

    def __repr__(self):
        return "OuterProductCell(*%r)" % list([self._A, self._B])


# --- Utility conversion functions

def as_cell(cell):
    """Convert any valid object to a Cell (in particular, cellname string),
    or return cell if it is already a Cell."""
    if isinstance(cell, Cell):
        return cell
    elif hasattr(cell, "ufl_cell"):
        return cell.ufl_cell()
    elif isinstance(cell, str):
        return Cell(cell)
    else:
        error("Invalid cell %s." % cell)
