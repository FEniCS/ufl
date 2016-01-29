# -*- coding: utf-8 -*-
"Types for representing a cell."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.utils.formatting import istr
from ufl.utils.dicts import EmptyDict
from ufl.core.terminal import Terminal
from ufl.core.ufl_type import attach_operators_from_hash_data


# Export list for ufl.classes
__all_classes__ = ["AbstractCell", "Cell", "TensorProductCell"]


# --- The most abstract cell class, base class for other cell types

class AbstractCell(object):
    "Representation of an abstract finite element cell with only the dimensions known."
    __slots__ = ("_topological_dimension", "_geometric_dimension")
    def __init__(self, topological_dimension, geometric_dimension):
        # Validate dimensions
        ufl_assert(isinstance(geometric_dimension, int),
                   "Expecting integer geometric dimension, not '%r'" % (geometric_dimension,))
        ufl_assert(isinstance(topological_dimension, int),
                   "Expecting integer topological dimension, not '%r'" % (topological_dimension,))
        ufl_assert(topological_dimension <= geometric_dimension,
                   "Topological dimension cannot be larger than geometric dimension.")

        # Store validated dimensions
        self._topological_dimension = topological_dimension
        self._geometric_dimension = geometric_dimension

    def topological_dimension(self):
        "Return the dimension of the topology of this cell."
        return self._topological_dimension

    def geometric_dimension(self):
        "Return the dimension of the space this cell is embedded in."
        return self._geometric_dimension

    def is_simplex(self):
        "Return True if this is a simplex cell."
        raise NotImplementedError("Implement this to allow important checks and optimizations.")

    def has_simplex_facets(self):
        "Return True if all the facets of this cell are simplex cells."
        raise NotImplementedError("Implement this to allow important checks and optimizations.")

    def __lt__(self, other):
        "Define an arbitrarily chosen but fixed sort order for all cells."
        if not isinstance(other, AbstractCell):
            return NotImplemented
        # Sort by gdim first, tdim next, then whatever's left depending on the subclass
        s = (self.geometric_dimension(), self.topological_dimension())
        o = (other.geometric_dimension(), other.topological_dimension())
        if s != o:
            return s < o
        return self._ufl_hash_data_() < other._ufl_hash_data_()


# --- Basic topological properties of known basic cells

# Mapping from cell name to number of cell entities of each topological dimension
num_cell_entities = {
    "vertex":        (1,),
    "interval":      (2,  1),
    "triangle":      (3,  3, 1),
    "quadrilateral": (4,  4, 1),
    "tetrahedron":   (4,  6, 4, 1),
    "hexahedron":    (8, 12, 6, 1),
    }

# Mapping from cell name to topological dimension
cellname2dim = dict((k, len(v)-1) for k,v in num_cell_entities.items())

# Mapping from cell name to facet name
# Note: This is not generalizable to product elements but it's still in use a couple of places.
cellname2facetname = {
    "interval":      "vertex",
    "triangle":      "interval",
    "quadrilateral": "interval",
    "tetrahedron":   "triangle",
    "hexahedron":    "quadrilateral",
    }


# --- Basic cell representation classes

@attach_operators_from_hash_data
class Cell(AbstractCell):
    "Representation of a named finite element cell with known structure."
    __slots__ = ("_cellname",)
    def __init__(self, cellname, geometric_dimension=None):
        "Initialize basic cell description."

        self._cellname = cellname

        # The topological dimension is defined by the cell type,
        # so the cellname must be among the known ones,
        # so we can find the known dimension, unless we have
        # a product cell, in which the given dimension is used
        topological_dimension = len(num_cell_entities[cellname]) - 1

        # The geometric dimension defaults to equal the topological
        # dimension unless overridden for embedded cells
        if geometric_dimension is None:
            geometric_dimension = topological_dimension

        # Initialize and validate dimensions
        AbstractCell.__init__(self, topological_dimension, geometric_dimension)

    # --- Overrides of AbstractCell methods ---

    def reconstruct(self, geometric_dimension=None):
        if geometric_dimension is None:
            geometric_dimension = self._geometric_dimension
        return Cell(self._cellname, geometric_dimension=geometric_dimension)

    def is_simplex(self):
        return self.num_vertices() == self.topological_dimension() + 1

    def has_simplex_facets(self):
        return self.is_simplex() or self.cellname() == "quadrilateral"

    # --- Specific cell properties ---

    def cellname(self):
        "Return the cellname of the cell."
        return self._cellname

    def num_vertices(self):
        "The number of cell vertices."
        return num_cell_entities[self.cellname()][0]

    def num_edges(self):
        "The number of cell edges."
        return num_cell_entities[self.cellname()][1]

    def num_facets(self):
        "The number of cell facets."
        tdim = self.topological_dimension()
        return num_cell_entities[self.cellname()][tdim-1]

    # --- Facet properties ---

    def num_facet_edges(self):
        "The number of facet edges."
        # This is used in geometry.py
        fn = cellname2facetname[self.cellname()]
        return num_cell_entities[fn][1]

    # --- Special functions for proper object behaviour ---

    def __str__(self):
        gdim = self.geometric_dimension()
        tdim = self.topological_dimension()
        s = self.cellname()
        if gdim > tdim:
            s += "%dD" % gdim
        return s

    def __repr__(self):
        # For standard cells, return name of builtin cell object if possible.
        # This reduces the size of the repr strings for domains, elements, etc. as well
        gdim = self.geometric_dimension()
        tdim = self.topological_dimension()
        name = self.cellname()
        if gdim == tdim and name in cellname2dim:
            return name
        else:
            return "Cell(%r, %r)" % (name, gdim)

    def _ufl_hash_data_(self):
        return (self._geometric_dimension, self._topological_dimension, self._cellname)


@attach_operators_from_hash_data
class TensorProductCell(AbstractCell):
    __slots__ = ("_cells",)

    def __init__(self, *cells, **kwargs):
        if kwargs and kwargs.keys() != ["geometric_dimension"]:
            raise TypeError(
                "TensorProductCell got an unexpected keyword argument '%s'" %
                kwargs.keys()[0])

        self._cells = tuple(as_cell(cell) for cell in cells)

        tdim = sum([cell.topological_dimension() for cell in self._cells])

        if kwargs:
            gdim = kwargs["geometric_dimension"]
        else:
            gdim = sum([cell.geometric_dimension() for cell in self._cells])

        AbstractCell.__init__(self, tdim, gdim)

    def is_simplex(self):
        "Return True if this is a simplex cell."
        if len(self._cells) == 1:
            return self._cells[0].is_simplex()
        return False

    def has_simplex_facets(self):
        "Return True if all the facets of this cell are simplex cells."
        if len(self._cells) == 1:
            return self._cells[0].has_simplex_facets()
        return False

    def num_vertices(self):
        "The number of cell vertices."
        return product(c.num_vertices() for c in self._cells)

    def num_edges(self):
        "The number of cell edges."
        error("Not defined for TensorProductCell.")

    def num_facets(self):
        "The number of cell facets."
        return sum(c.num_facets() for c in self._cells if c.topological_dimension() > 0)

    def sub_cells(self):
        "Return list of cell factors."
        return self._cells

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "TensorProductCell(%s)" % ", ".join(repr(c) for c in self._cells)

    def _ufl_hash_data_(self):
        return tuple(c._ufl_hash_data_() for c in self._cells)


# --- Utility conversion functions

# Mapping from topological dimension to reference cell name for simplices
_simplex_dim2cellname = {
    0: "vertex",
    1: "interval",
    2: "triangle",
    3: "tetrahedron",
    }

# Mapping from topological dimension to reference cell name for hypercubes
_hypercube_dim2cellname = {
    0: "vertex",
    1: "interval",
    2: "quadrilateral",
    3: "hexahedron",
    }

def simplex(topological_dimension, geometric_dimension=None):
    "Return a simplex cell of given dimension."
    return Cell(_simplex_dim2cellname[topological_dimension], geometric_dimension)

def hypercube(topological_dimension, geometric_dimension=None):
    "Return a hypercube cell of given dimension."
    return Cell(_hypercube_dim2cellname[topological_dimension], geometric_dimension)

def as_cell(cell):
    """Convert any valid object to a Cell or return cell if it is already a Cell.

    Allows an already valid cell, a known cellname string, or a tuple of cells for a product cell.
    """
    if isinstance(cell, AbstractCell):
        return cell
    elif isinstance(cell, str):
        return Cell(cell)
    elif isinstance(cell, tuple):
        return TensorProductCell(cell)
    else:
        error("Invalid cell %s." % cell)
