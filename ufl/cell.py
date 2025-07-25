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
import weakref
from abc import abstractmethod

from ufl.core.ufl_type import UFLObject

__all_classes__ = ["AbstractCell", "Cell", "TensorProductCell"]


class AbstractCell(UFLObject):
    """A base class for all cells."""

    @abstractmethod
    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""

    @abstractmethod
    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""

    @abstractmethod
    def has_simplex_facets(self) -> bool:
        """Return True if all the facets of this cell are simplex cells."""

    @abstractmethod
    def _lt(self, other) -> bool:
        """Less than operator.

        Define an arbitrarily chosen but fixed sort order for all
        instances of this type with the same dimensions.
        """

    @abstractmethod
    def num_sub_entities(self, dim: int) -> int:
        """Get the number of sub-entities of the given dimension."""

    @abstractmethod
    def sub_entities(self, dim: int) -> tuple[AbstractCell, ...]:
        """Get the sub-entities of the given dimension."""

    @abstractmethod
    def sub_entity_types(self, dim: int) -> tuple[AbstractCell, ...]:
        """Get the unique sub-entity types of the given dimension."""

    @abstractmethod
    def cellname(self) -> str:
        """Return the cellname of the cell."""

    @abstractmethod
    def reconstruct(self, **kwargs: typing.Any) -> AbstractCell:
        """Reconstruct this cell, overwriting properties by those in kwargs."""

    def __lt__(self, other: AbstractCell) -> bool:
        """Define an arbitrarily chosen but fixed sort order for all cells."""
        if type(self) is type(other):
            s = self.topological_dimension()
            o = other.topological_dimension()
            if s != o:
                return s < o
            return self._lt(other)
        else:
            if type(self).__name__ == type(other).__name__:
                raise ValueError("Cannot order cell types with the same name")
            return type(self).__name__ < type(other).__name__

    def num_vertices(self) -> int:
        """Get the number of vertices."""
        return self.num_sub_entities(0)

    def num_edges(self) -> int:
        """Get the number of edges."""
        return self.num_sub_entities(1)

    def num_faces(self) -> int:
        """Get the number of faces."""
        return self.num_sub_entities(2)

    def num_facets(self) -> int:
        """Get the number of facets.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension()
        return self.num_sub_entities(tdim - 1)

    def num_ridges(self) -> int:
        """Get the number of ridges.

        Ridges are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension()
        return self.num_sub_entities(tdim - 2)

    def num_peaks(self) -> int:
        """Get the number of peaks.

        Peaks are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension()
        return self.num_sub_entities(tdim - 3)

    def vertices(self) -> tuple[AbstractCell, ...]:
        """Get the vertices."""
        return self.sub_entities(0)

    def edges(self) -> tuple[AbstractCell, ...]:
        """Get the edges."""
        return self.sub_entities(1)

    def faces(self) -> tuple[AbstractCell, ...]:
        """Get the faces."""
        return self.sub_entities(2)

    def facets(self) -> tuple[AbstractCell, ...]:
        """Get the facets.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension()
        return self.sub_entities(tdim - 1)

    def ridges(self) -> tuple[AbstractCell, ...]:
        """Get the ridges.

        Ridges are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension()
        return self.sub_entities(tdim - 2)

    def peaks(self) -> tuple[AbstractCell, ...]:
        """Get the peaks.

        Peaks are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension()
        return self.sub_entities(tdim - 3)

    def vertex_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique vertices types."""
        return self.sub_entity_types(0)

    def edge_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique edge types."""
        return self.sub_entity_types(1)

    def face_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique face types."""
        return self.sub_entity_types(2)

    def facet_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique facet types.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension()
        return self.sub_entity_types(tdim - 1)

    def ridge_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique ridge types.

        Ridges are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension()
        return self.sub_entity_types(tdim - 2)

    def peak_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique peak types.

        Peaks are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension()
        return self.sub_entity_types(tdim - 3)


_sub_entity_celltypes: dict[str, list[tuple[str, ...]]] = {
    "vertex": [("vertex",)],
    "interval": [tuple("vertex" for i in range(2)), ("interval",)],
    "triangle": [
        tuple("vertex" for i in range(3)),
        tuple("interval" for i in range(3)),
        ("triangle",),
    ],
    "quadrilateral": [
        tuple("vertex" for i in range(4)),
        tuple("interval" for i in range(4)),
        ("quadrilateral",),
    ],
    "tetrahedron": [
        tuple("vertex" for i in range(4)),
        tuple("interval" for i in range(6)),
        tuple("triangle" for i in range(4)),
        ("tetrahedron",),
    ],
    "hexahedron": [
        tuple("vertex" for i in range(8)),
        tuple("interval" for i in range(12)),
        tuple("quadrilateral" for i in range(6)),
        ("hexahedron",),
    ],
    "prism": [
        tuple("vertex" for i in range(6)),
        tuple("interval" for i in range(9)),
        ("triangle", "quadrilateral", "quadrilateral", "quadrilateral", "triangle"),
        ("prism",),
    ],
    "pyramid": [
        tuple("vertex" for i in range(5)),
        tuple("interval" for i in range(8)),
        ("quadrilateral", "triangle", "triangle", "triangle", "triangle"),
        ("pyramid",),
    ],
    "pentatope": [
        tuple("vertex" for i in range(5)),
        tuple("interval" for i in range(10)),
        tuple("triangle" for i in range(10)),
        tuple("tetrahedron" for i in range(5)),
        ("pentatope",),
    ],
    "tesseract": [
        tuple("vertex" for i in range(16)),
        tuple("interval" for i in range(32)),
        tuple("quadrilateral" for i in range(24)),
        tuple("hexahedron" for i in range(8)),
        ("tesseract",),
    ],
}


class Cell(AbstractCell):
    """Representation of a named finite element cell with known structure."""

    __slots__ = (
        "_cellname",
        "_num_cell_entities",
        "_sub_entities",
        "_sub_entity_types",
        "_sub_entity_types",
        "_tdim",
    )

    def __init__(self, cellname: str):
        """Initialise.

        Args:
            cellname: Name of the cell
        """
        if cellname not in _sub_entity_celltypes:
            raise ValueError(f"Unsupported cell type: {cellname}")

        self._sub_entity_celltypes = _sub_entity_celltypes[cellname]

        self._cellname = cellname
        self._tdim = len(self._sub_entity_celltypes) - 1

        self._num_cell_entities = [len(i) for i in self._sub_entity_celltypes]
        self._sub_entities = [
            tuple(Cell(t) for t in se_types) for se_types in self._sub_entity_celltypes[:-1]
        ]
        self._sub_entity_types = [tuple(set(i)) for i in self._sub_entities]
        self._sub_entities.append((weakref.proxy(self),))
        self._sub_entity_types.append((weakref.proxy(self),))

        if not isinstance(self._tdim, numbers.Integral):
            raise ValueError("Expecting integer topological_dimension.")

    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""
        return self._tdim

    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""
        return self._cellname in ["vertex", "interval", "triangle", "tetrahedron"]

    def has_simplex_facets(self) -> bool:
        """Return True if all the facets of this cell are simplex cells."""
        return self._cellname in ["interval", "triangle", "quadrilateral", "tetrahedron"]

    def num_sub_entities(self, dim: int) -> int:
        """Get the number of sub-entities of the given dimension."""
        if dim < 0:
            return 0
        try:
            return self._num_cell_entities[dim]
        except IndexError:
            return 0

    def sub_entities(self, dim: int) -> tuple[AbstractCell, ...]:
        """Get the sub-entities of the given dimension."""
        if dim < 0:
            return ()
        try:
            return self._sub_entities[dim]
        except IndexError:
            return ()

    def sub_entity_types(self, dim: int) -> tuple[AbstractCell, ...]:
        """Get the unique sub-entity types of the given dimension."""
        if dim < 0:
            return ()
        try:
            return self._sub_entity_types[dim]
        except IndexError:
            return ()

    def _lt(self, other) -> bool:
        return self._cellname < other._cellname

    def cellname(self) -> str:
        """Return the cellname of the cell."""
        return self._cellname

    def __str__(self) -> str:
        """Format as a string."""
        return self._cellname

    def __repr__(self) -> str:
        """Representation."""
        return self._cellname

    def _ufl_hash_data_(self) -> typing.Hashable:
        """UFL hash data."""
        return (self._cellname,)

    def reconstruct(self, **kwargs: typing.Any) -> Cell:
        """Reconstruct this cell, overwriting properties by those in kwargs."""
        for key, value in kwargs.items():
            raise TypeError(f"reconstruct() got unexpected keyword argument '{key}'")
        return Cell(self._cellname)


class TensorProductCell(AbstractCell):
    """Tensor product cell."""

    __slots__ = ("_cells", "_tdim")

    def __init__(self, *cells: AbstractCell):
        """Initialise.

        Args:
            cells: Cells to take the tensor product of
        """
        self._cells = tuple(as_cell(cell) for cell in cells)

        self._tdim = sum([cell.topological_dimension() for cell in self._cells])

        if not isinstance(self._tdim, numbers.Integral):
            raise ValueError("Expecting integer topological_dimension.")

    def sub_cells(self) -> tuple[AbstractCell, ...]:
        """Return list of cell factors."""
        return self._cells

    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""
        return self._tdim

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
            # Note: This is not the number of facets that the cell has,
            # but I'm leaving it here for now to not change past
            # behaviour
            return sum(c.num_facets() for c in self._cells if c.topological_dimension() > 0)
        if dim == self._tdim:
            return 1
        raise NotImplementedError(f"TensorProductCell.num_sub_entities({dim}) is not implemented.")

    def sub_entities(self, dim: int) -> tuple[AbstractCell, ...]:
        """Get the sub-entities of the given dimension."""
        if dim < 0 or dim > self._tdim:
            return ()
        if dim == 0:
            return tuple(Cell("vertex") for i in range(self.num_sub_entities(0)))
        if dim == self._tdim:
            return (self,)
        raise NotImplementedError(f"TensorProductCell.sub_entities({dim}) is not implemented.")

    def sub_entity_types(self, dim: int) -> tuple[AbstractCell, ...]:
        """Get the unique sub-entity types of the given dimension."""
        if dim < 0 or dim > self._tdim:
            return ()
        if dim == 0:
            return (Cell("vertex"),)
        if dim == self._tdim:
            return (self,)
        raise NotImplementedError(f"TensorProductCell.sub_entities({dim}) is not implemented.")

    def _lt(self, other) -> bool:
        return self._ufl_hash_data_() < other._ufl_hash_data_()

    def cellname(self) -> str:
        """Return the cellname of the cell."""
        return " * ".join([cell.cellname() for cell in self._cells])

    def __str__(self) -> str:
        """Format as a string."""
        return "TensorProductCell(" + ", ".join(f"{c!r}" for c in self._cells) + ")"

    def __repr__(self) -> str:
        """Representation."""
        return str(self)

    def _ufl_hash_data_(self) -> typing.Hashable:
        """UFL hash data."""
        return tuple(c._ufl_hash_data_() for c in self._cells)

    def reconstruct(self, **kwargs: typing.Any) -> AbstractCell:
        """Reconstruct this cell, overwriting properties by those in kwargs."""
        for key, value in kwargs.items():
            raise TypeError(f"reconstruct() got unexpected keyword argument '{key}'")
        return TensorProductCell(*self._cells)


def simplex(topological_dimension: int):
    """Return a simplex cell of the given dimension."""
    if topological_dimension == 0:
        return Cell("vertex")
    if topological_dimension == 1:
        return Cell("interval")
    if topological_dimension == 2:
        return Cell("triangle")
    if topological_dimension == 3:
        return Cell("tetrahedron")
    if topological_dimension == 4:
        return Cell("pentatope")
    raise ValueError(f"Unsupported topological dimension for simplex: {topological_dimension}")


def hypercube(topological_dimension: int):
    """Return a hypercube cell of the given dimension."""
    if topological_dimension == 0:
        return Cell("vertex")
    if topological_dimension == 1:
        return Cell("interval")
    if topological_dimension == 2:
        return Cell("quadrilateral")
    if topological_dimension == 3:
        return Cell("hexahedron")
    if topological_dimension == 4:
        return Cell("tesseract")
    raise ValueError(f"Unsupported topological dimension for hypercube: {topological_dimension}")


def as_cell(cell: AbstractCell | str | tuple[AbstractCell, ...]) -> AbstractCell:
    """Convert any valid object to a Cell or return cell if it is already a Cell.

    Allows an already valid cell, a known cellname string, or a tuple of cells for a product cell.
    """
    if isinstance(cell, AbstractCell):
        return cell
    elif isinstance(cell, str):
        return Cell(cell)
    elif isinstance(cell, tuple):
        return TensorProductCell(*cell)
    else:
        raise ValueError(f"Invalid cell {cell}.")
