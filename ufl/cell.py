"""Types for representing a cell."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations
import functools
import numbers
import typing

from ufl.core.ufl_type import UFLObject
from abc import abstractmethod

try:
    from typing import Self
except ImportError:
    # Hack to get this to work pre Python 3.11
    # Delete this after 04 Oct 2026 (Python 3.10 end of life)
    # After this date, can also replace "self: Self," with "self,
    from typing import TypeVar
    Self = TypeVar("Self", bound="AbstractCellBase")

__all_classes__ = ["AbstractCellBase", "AbstractCell", "CellBase", "Cell", "TensorProductCell"]


class AbstractCellBase(UFLObject):
    """A base class for all cells that allows for abstract cells where only the dimensions are known."""
    @abstractmethod
    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""

    @abstractmethod
    def geometric_dimension(self) -> int:
        """Return the dimension of the geometry of this cell."""

    @abstractmethod
    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""

    @abstractmethod
    def has_simplex_facets(self) -> bool:
        """Return True if all the facets of this cell are simplex cells."""

    @abstractmethod
    def _lt(self: Self, other: Self) -> bool:
        """Define an arbitrarily chosen but fixed sort order for all instances of this type with the same dimensions."""

    def __lt__(self, other: AbstractCellBase) -> bool:
        """Define an arbitrarily chosen but fixed sort order for all cells."""
        if type(self) == type(other):
            s = (self.geometric_dimension(), self.topological_dimension())
            o = (other.geometric_dimension(), other.topological_dimension())
            if s != o:
                return s < o
            return self._lt(other)
        else:
            if type(self).__name__ == type(other).__name__:
                raise ValueError("Cannot order cell types with the same name")
            return type(self).__name__ < type(other).__name__


class CellBase(AbstractCellBase):
    """A base class for all cells."""

    @abstractmethod
    def num_sub_entities(self, dim: int) -> int:
        """Get the number of sub-entities of the given dimension."""

    @abstractmethod
    def sub_entities(self, dim: int) -> typing.List[CellBase]:
        """Get the sub-entities of the given dimension."""

    @abstractmethod
    def sub_entity_types(self, dim: int) -> typing.List[CellBase]:
        """Get the unique sub-entity types of the given dimension."""

    @abstractmethod
    def cellname(self) -> str:
        """Return the cellname of the cell."""

    def num_vertices(self) -> int:
        """Get the number of vertices."""
        return self.num_sub_entities(0)

    def num_edges(self) -> int:
        """Get the number of edges."""
        return self.num_sub_entities(1)

    def num_faces(self) -> int:
        """Get the number of faces."""
        return self.num_sub_entities(2)

    def num_volumes(self) -> int:
        """Get the number of faces."""
        return self.num_sub_entities(3)

    def num_facets(self) -> int:
        """Get the number of facets.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension()
        return self.num_sub_entities(tdim - 1)

    def num_ridges(self) -> int:
        """Get the number of ridges.

        Facets are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension()
        return self.num_sub_entities(tdim - 2)

    def num_peaks(self) -> int:
        """Get the number of peaks.

        Facets are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension()
        return self.num_sub_entities(tdim - 3)

    def vertices(self) -> typing.List[CellBase]:
        """Get the vertices."""
        return self.sub_entities(0)

    def edges(self) -> typing.List[CellBase]:
        """Get the edges."""
        return self.sub_entities(1)

    def faces(self) -> typing.List[CellBase]:
        """Get the faces."""
        return self.sub_entities(2)

    def volumes(self) -> typing.List[CellBase]:
        """Get the faces."""
        return self.sub_entities(3)

    def facets(self) -> typing.List[CellBase]:
        """Get the facets.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension()
        return self.sub_entities(tdim - 1)

    def ridges(self) -> typing.List[CellBase]:
        """Get the ridges.

        Facets are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension()
        return self.sub_entities(tdim - 2)

    def peaks(self) -> typing.List[CellBase]:
        """Get the peaks.

        Facets are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension()
        return self.sub_entities(tdim - 3)

    def vertex_types(self) -> typing.List[CellBase]:
        """Get the unique vertices types."""
        return self.sub_entity_types(0)

    def edge_types(self) -> typing.List[CellBase]:
        """Get the unique edge types."""
        return self.sub_entity_types(1)

    def face_types(self) -> typing.List[CellBase]:
        """Get the unique face types."""
        return self.sub_entity_types(2)

    def volume_types(self) -> typing.List[CellBase]:
        """Get the unique face types."""
        return self.sub_entity_types(3)

    def facet_types(self) -> typing.List[CellBase]:
        """Get the unique facet types.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension()
        return self.sub_entity_types(tdim - 1)

    def ridge_types(self) -> typing.List[CellBase]:
        """Get the unique ridge types.

        Facets are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension()
        return self.sub_entity_types(tdim - 2)

    def peak_types(self) -> typing.List[CellBase]:
        """Get the unique peak types.

        Facets are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension()
        return self.sub_entity_types(tdim - 3)


class AbstractCell(AbstractCellBase):
    """Representation of an abstract finite element cell with only the dimensions known."""
    __slots__ = ("_tdim", "_gdim")

    def __init__(self, topological_dimension: int, geometric_dimension: int):
        # Validate dimensions
        if not isinstance(geometric_dimension, numbers.Integral):
            raise ValueError("Expecting integer geometric_dimension.")
        if not isinstance(topological_dimension, numbers.Integral):
            raise ValueError("Expecting integer topological_dimension.")
        if topological_dimension > geometric_dimension:
            raise ValueError("Topological dimension cannot be larger than geometric dimension.")

        # Store validated dimensions
        self._tdim = topological_dimension
        self._gdim = geometric_dimension

    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""
        return self._tdim

    def geometric_dimension(self) -> int:
        """Return the dimension of the geometry of this cell."""
        return self._gdim

    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""
        raise NotImplementedError("Implement this to allow important checks and optimizations.")

    def has_simplex_facets(self) -> bool:
        """Return True if all the facets of this cell are simplex cells."""
        raise NotImplementedError("Implement this to allow important checks and optimizations.")

    def _lt(self: Self, other: Self) -> bool:
        # Sort by gdim first, tdim next, then whatever's left depending on the subclass
        return False

    def _ufl_has_data_(self):
        return (self._tdim, self._gdim)

    def __str__(self):
        return f"AbstractCell({self._tdim}, {self._gdim})"

    def __repr__(self):
        return f"AbstractCell({self._tdim}, {self._gdim})"


class Cell(CellBase):
    """Representation of a named finite element cell with known structure."""
    __slots__ = ("_cellname", "_tdim", "_gdim", "_num_cell_entities", "_sub_entity_types")

    def __init__(self, cellname: str, geometric_dimension: typing.Optional[int] = None):
        if cellname == "vertex":
            self._num_cell_entities = (1, )
            self._sub_entity_types = [["vertex"]]
        elif cellname == "interval":
            self._num_cell_entities = (2, 1)
            self._sub_entity_types = [["vertex" for i in range(2)], ["interval"]]
        elif cellname == "triangle":
            self._num_cell_entities = (3, 3, 1)
            self._sub_entity_types = [["vertex" for i in range(3)], ["interval" for i in range(3)], ["triangle"]]
        elif cellname == "quadrilateral":
            self._num_cell_entities = (4, 4, 1)
            self._sub_entity_types = [["vertex" for i in range(4)], ["interval" for i in range(4)], ["quadrilateral"]]
        elif cellname == "tetrahedron":
            self._num_cell_entities = (4, 6, 4, 1)
            self._sub_entity_types = [["vertex" for i in range(4)], ["interval" for i in range(4)], ["triangle" for i in range(4)], ["tetrahedron"]]
        elif cellname == "prism":
            self._num_cell_entities = (6, 9, 5, 1)
            self._sub_entity_types = [["vertex" for i in range(6)], ["interval" for i in range(9)], ["triangle", "quadrilateral", "quadrilateral", "quadrilateral", "triangle"], ["prism"]]
        elif cellname == "pyramid":
            self._num_cell_entities = (5, 8, 5, 1)
            self._sub_entity_types = [["vertex" for i in range(5)], ["interval" for i in range(8)], ["quadrilateral", "triangle", "triangle", "triangle", "triangle"], ["pyramid"]]
        elif cellname == "hexahedron":
            self._num_cell_entities = (8, 12, 6, 1)
            self._sub_entity_types = [["vertex" for i in range(8)], ["interval" for i in range(12)], ["quadrilateral" for i in range(6)], ["hexahedron"]]
        else:
            raise ValueError(f"Unsupported cell type: {cellname}")

        self._cellname = cellname
        self._tdim = len(self._num_cell_entities) - 1
        self._gdim = self._tdim if geometric_dimension is None else geometric_dimension

        if not isinstance(self._gdim, numbers.Integral):
            raise ValueError("Expecting integer geometric_dimension.")
        if not isinstance(self._tdim, numbers.Integral):
            raise ValueError("Expecting integer topological_dimension.")
        if self._tdim > self._gdim:
            raise ValueError("Topological dimension cannot be larger than geometric dimension.")

    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""
        return self._tdim

    def geometric_dimension(self) -> int:
        """Return the dimension of the geometry of this cell."""
        return self._gdim

    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""
        print(self._cellname)
        return self._cellname in ["vertex", "interval", "triangle", "tetrahedron"]

    def has_simplex_facets(self) -> bool:
        """Return True if all the facets of this cell are simplex cells."""
        return self._cellname in ["interval", "triangle", "quadrilateral", "tetrahedron"]

    def num_sub_entities(self, dim: int) -> int:
        """Get the number of sub-entities of the given dimension."""
        try:
            return self._num_cell_entities[dim]
        except IndexError:
            return 0

    def sub_entities(self, dim: int) -> typing.List[CellBase]:
        """Get the sub-entities of the given dimension."""
        try:
            return [Cell(t, self._gdim) for t in self._subentity_types[dim]]
        except IndexError:
            return 0

    def sub_entity_types(self, dim: int) -> typing.List[CellBase]:
        """Get the unique sub-entity types of the given dimension."""
        try:
            return [Cell(t, self._gdim) for t in set(self._subentity_types[dim])]
        except IndexError:
            return 0

    def _lt(self: Self, other: Self) -> bool:
        return self._cellname < other._cellname

    def cellname(self) -> str:
        """Return the cellname of the cell."""
        return self._cellname

    def __str__(self) -> str:
        if self._gdim == self._tdim:
            return self._cellname
        else:
            return f"{self._cellname}{self._gdim}D"

    def __repr__(self) -> str:
        if self._gdim == self._tdim:
            return self._cellname
        else:
            return f"Cell({self._cellname}, {self._gdim})"

    def _ufl_hash_data_(self) -> typing.Hashable:
        return (self._cellname, self._gdim)

    # TODO: put this in the base class?
    def reconstruct(self, geometric_dimension: typing.Optional[int] = None) -> Cell:
        """Reconstruct this cell."""
        return Cell(self._cellname, geometric_dimension=self._gdim if geometric_dimension is None else geometric_dimension)


class TensorProductCell(CellBase):
    __slots__ = ("_cells", "_tdim", "_gdim")

    def __init__(self, *cells, geometric_dimension: typing.Optional[int] = None):
        self._cells = tuple(as_cell(cell) for cell in cells)

        self._tdim = sum([cell.topological_dimension() for cell in self._cells])
        self._gdim = self._tdim if geometric_dimension is None else geometric_dimension

        if not isinstance(self._gdim, numbers.Integral):
            raise ValueError("Expecting integer geometric_dimension.")
        if not isinstance(self._tdim, numbers.Integral):
            raise ValueError("Expecting integer topological_dimension.")
        if self._tdim > self._gdim:
            raise ValueError("Topological dimension cannot be larger than geometric dimension.")

    # TODO: put this in the base class?
    def sub_cells(self) -> typing.List[CellBase]:
        """Return list of cell factors."""
        return self._cells

    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""
        return self._tdim

    def geometric_dimension(self) -> int:
        """Return the dimension of the geometry of this cell."""
        return self._gdim

    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""
        if len(self._cells) == 1:
            return self._cells[0].is_simplex()
        return False

    def has_simplex_facets(self) -> bool:
        """Return True if all the facets of this cell are simplex cells."""
        if len(self._cells) == 1:
            return self._cells[0].has_simplex_facets()
        if self._tdim == 1:
            return True
        return False

    def num_sub_entities(self, dim: int) -> int:
        """Get the number of sub-entities of the given dimension."""
        if dim < 0 or dim > self._tdim:
            return 0
        if dim == 0:
            return functools.reduce(lambda x, y: x * y, [c.num_vertices() for c in self._cells])
        if dim == self._tdim - 1:
            # Note: This is not the number of facets that the cell has, but I'm leaving it here for now
            # to not change past behaviour
            return sum(c.num_facets() for c in self._cells if c.topological_dimension() > 0)
        if dim == self._tdim:
            return 1
        raise NotImplementedError(f"TensorProductCell.num_sub_entities({dim}) is not implemented.")

    def sub_entities(self, dim: int) -> typing.List[CellBase]:
        """Get the sub-entities of the given dimension."""
        if dim < 0 or dim > self._tdim:
            return []
        if dim == 0:
            return [Cell("vertex", self._gdim) for i in range(self.num_sub_entities(0))]
        if dim == self._tdim:
            return [self]
        raise NotImplementedError(f"TensorProductCell.sub_entities({dim}) is not implemented.")

    def sub_entity_types(self, dim: int) -> typing.List[CellBase]:
        """Get the unique sub-entity types of the given dimension."""
        if dim < 0 or dim > self._tdim:
            return []
        if dim == 0:
            return [Cell("vertex", self._gdim)]
        if dim == self._tdim:
            return [self]
        raise NotImplementedError(f"TensorProductCell.sub_entities({dim}) is not implemented.")

    def _lt(self: Self, other: Self) -> bool:
        return self._ufl_hash_data_() < other._ufl_hash_data_()

    def cellname(self) -> str:
        """Return the cellname of the cell."""
        return " * ".join([cell.cellname() for cell in self._cells])

    def __str__(self) -> str:
        s = "TensorProductCell("
        s += ", ".join(f"{c!r}" for c in self._cells)
        if self._tdim != self._gdim:
            s += f", geometric_dimension={self._gdim}"
        s += ")"
        return s

    def __repr__(self) -> str:
        return str(self)

    def _ufl_hash_data_(self) -> typing.Hashable:
        return tuple(c._ufl_hash_data_() for c in self._cells) + (self._gdim,)

    def reconstruct(self, geometric_dimension: typing.Optional[int] = None) -> Cell:
        """Reconstruct this cell."""
        return TensorProductCell(self._cellname, geometric_dimension=self._gdim if geometric_dimension is None else geometric_dimension)


def simplex(topological_dimension: int, geometric_dimension: typing.Optional[int] = None):
    """Return a simplex cell of the given dimension."""
    if topological_dimension == 0:
        return Cell("vertex", geometric_dimension)
    if topological_dimension == 1:
        return Cell("interval", geometric_dimension)
    if topological_dimension == 2:
        return Cell("triangle", geometric_dimension)
    if topological_dimension == 3:
        return Cell("tetrahedron", geometric_dimension)
    raise ValueError(f"Unsupported topological dimension for simplex: {topological_dimension}")


def hypercube(topological_dimension, geometric_dimension=None):
    """Return a hypercube cell of the given dimension."""
    if topological_dimension == 0:
        return Cell("vertex", geometric_dimension)
    if topological_dimension == 1:
        return Cell("interval", geometric_dimension)
    if topological_dimension == 2:
        return Cell("quadrilateral", geometric_dimension)
    if topological_dimension == 3:
        return Cell("hexahedron", geometric_dimension)
    raise ValueError(f"Unsupported topological dimension for hypercube: {topological_dimension}")


def as_cell(cell: typing.Union[AbstractCellBase, str, typing.Tuple[AbstractCellBase, ...]]) -> AbstractCellBase:
    """Convert any valid object to a Cell or return cell if it is already a Cell.

    Allows an already valid cell, a known cellname string, or a tuple of cells for a product cell.
    """
    if isinstance(cell, AbstractCellBase):
        return cell
    elif isinstance(cell, str):
        return Cell(cell)
    elif isinstance(cell, tuple):
        return TensorProductCell(cell)
    else:
        raise ValueError(f"Invalid cell {cell}.")
