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
from collections.abc import Sequence

from ufl.core.ufl_type import UFLObject

__all_classes__ = ["AbstractCell", "Cell", "TensorProductCell"]


class AbstractCell(UFLObject):
    """A base class for all cells."""

    @property
    @abstractmethod
    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""

    @property
    @abstractmethod
    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""

    @property
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

    @property
    @abstractmethod
    def cellname(self) -> str:
        """Return the cellname of the cell."""

    @abstractmethod
    def reconstruct(self, **kwargs: typing.Any) -> AbstractCell:
        """Reconstruct this cell, overwriting properties by those in kwargs."""

    def __lt__(self, other: AbstractCell) -> bool:
        """Define an arbitrarily chosen but fixed sort order for all cells."""
        if type(self) is type(other):
            s = self.topological_dimension
            o = other.topological_dimension
            if s != o:
                return s < o
            return self._lt(other)
        else:
            if type(self).__name__ == type(other).__name__:
                raise ValueError("Cannot order cell types with the same name")
            return type(self).__name__ < type(other).__name__

    @property
    def num_vertices(self) -> int:
        """Get the number of vertices."""
        return self.num_sub_entities(0)

    @property
    def num_edges(self) -> int:
        """Get the number of edges."""
        return self.num_sub_entities(1)

    @property
    def num_faces(self) -> int:
        """Get the number of faces."""
        return self.num_sub_entities(2)

    @property
    def num_facets(self) -> int:
        """Get the number of facets.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension
        return self.num_sub_entities(tdim - 1)

    @property
    def num_ridges(self) -> int:
        """Get the number of ridges.

        Ridges are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension
        return self.num_sub_entities(tdim - 2)

    @property
    def num_peaks(self) -> int:
        """Get the number of peaks.

        Peaks are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension
        return self.num_sub_entities(tdim - 3)

    @property
    def vertices(self) -> tuple[AbstractCell, ...]:
        """Get the vertices."""
        return self.sub_entities(0)

    @property
    def edges(self) -> tuple[AbstractCell, ...]:
        """Get the edges."""
        return self.sub_entities(1)

    @property
    def faces(self) -> tuple[AbstractCell, ...]:
        """Get the faces."""
        return self.sub_entities(2)

    @property
    def facets(self) -> tuple[AbstractCell, ...]:
        """Get the facets.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension
        return self.sub_entities(tdim - 1)

    @property
    def ridges(self) -> tuple[AbstractCell, ...]:
        """Get the ridges.

        Ridges are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension
        return self.sub_entities(tdim - 2)

    @property
    def peaks(self) -> tuple[AbstractCell, ...]:
        """Get the peaks.

        Peaks are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension
        return self.sub_entities(tdim - 3)

    @property
    def vertex_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique vertices types."""
        return self.sub_entity_types(0)

    @property
    def edge_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique edge types."""
        return self.sub_entity_types(1)

    @property
    def face_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique face types."""
        return self.sub_entity_types(2)

    @property
    def facet_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique facet types.

        Facets are entities of dimension tdim-1.
        """
        tdim = self.topological_dimension
        return self.sub_entity_types(tdim - 1)

    @property
    def ridge_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique ridge types.

        Ridges are entities of dimension tdim-2.
        """
        tdim = self.topological_dimension
        return self.sub_entity_types(tdim - 2)

    @property
    def peak_types(self) -> tuple[AbstractCell, ...]:
        """Get the unique peak types.

        Peaks are entities of dimension tdim-3.
        """
        tdim = self.topological_dimension
        return self.sub_entity_types(tdim - 3)

    @property
    @abstractmethod
    def cells(self) -> tuple[AbstractCell, ...]:
        """Return the component cells in the CellSequence sense."""


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

    @property
    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""
        return self._tdim

    @property
    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""
        return self._cellname in ["vertex", "interval", "triangle", "tetrahedron"]

    @property
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

    @property
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

    @property
    def cells(self) -> tuple[AbstractCell, ...]:
        """Return the component cells in the CellSequence sense."""
        return (self,)


class TensorProductCell(AbstractCell):
    """Tensor product cell."""

    __slots__ = ("_cells", "_tdim")

    def __init__(self, *cells: AbstractCell):
        """Initialise.

        Args:
            cells: Cells to take the tensor product of
        """
        self._cells = tuple(as_cell(cell) for cell in cells)

        self._tdim = sum([cell.topological_dimension for cell in self._cells])

        if not isinstance(self._tdim, numbers.Integral):
            raise ValueError("Expecting integer topological_dimension.")

    @property
    def sub_cells(self) -> tuple[AbstractCell, ...]:
        """Return list of cell factors."""
        return self._cells

    @property
    def topological_dimension(self) -> int:
        """Return the dimension of the topology of this cell."""
        return self._tdim

    @property
    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""
        if len(self._cells) == 1:
            return self._cells[0].is_simplex
        return False

    @property
    def has_simplex_facets(self) -> bool:
        """Return True if all the facets of this cell are simplex cells."""
        if len(self._cells) == 1:
            return self._cells[0].has_simplex_facets
        if self._tdim == 1:
            return True
        return False

    def num_sub_entities(self, dim: int) -> int:
        """Get the number of sub-entities of the given dimension."""
        if dim < 0 or dim > self._tdim:
            return 0
        if dim == 0:
            return functools.reduce(lambda x, y: x * y, [c.num_vertices for c in self._cells])
        if dim == self._tdim - 1:
            # Note: This is not the number of facets that the cell has,
            # but I'm leaving it here for now to not change past
            # behaviour
            return sum(c.num_facets for c in self._cells if c.topological_dimension > 0)
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

    @property
    def cellname(self) -> str:
        """Return the cellname of the cell."""
        return " * ".join([cell.cellname for cell in self._cells])

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

    @property
    def cells(self) -> tuple[AbstractCell, ...]:
        """Return the component cells in the CellSequence sense."""
        return (self,)


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


class CellSequence(AbstractCell):
    """Representation of a sequence of cells."""

    __slots__ = (
        "_cells",
        "_tdim",
    )

    def __init__(self, cells: Sequence[AbstractCell]):
        """Initialise.

        Args:
            cells: Sequence of cells
        """
        self._cells = tuple(cells)
        self._tdim = max(cell.topological_dimension for cell in cells)

    @property
    def topological_dimension(self) -> int:
        """Return the maximum dimension of the component cells."""
        return self._tdim

    @property
    def is_simplex(self) -> bool:
        """Return True if this is a simplex cell."""
        raise RuntimeError(f"is_simplex not defined for {type(self).__name__}")

    @property
    def has_simplex_facets(self) -> bool:
        """Return True if all the facets of this cell are simplex cells."""
        raise RuntimeError(f"has_simplex_facets not defined for {type(self).__name__}")

    def num_sub_entities(self, dim: int) -> int:
        """Get the number of sub-entities of the given dimension."""
        raise RuntimeError(f"num_sub_entities not defined for {type(self).__name__}")

    def sub_entities(self, dim: int) -> tuple[AbstractCell, ...]:
        """Get the sub-entities of the given dimension."""
        raise RuntimeError(f"sub_entities not defined for {type(self).__name__}")

    def sub_entity_types(self, dim: int) -> tuple[AbstractCell, ...]:
        """Get the unique sub-entity types of the given dimension."""
        raise RuntimeError(f"sub_entity_types not defined for {type(self).__name__}")

    def _lt(self, other) -> bool:
        raise RuntimeError(f"_lt not defined for {type(self).__name__}")

    @property
    def cellname(self) -> str:
        """Return the cellname of the cell."""
        return f"{type(self).__name__}({tuple(c.cellname for c in self._cells)})"

    def reconstruct(self, **kwargs: typing.Any) -> CellSequence:
        """Reconstruct this cell, overwriting properties by those in kwargs."""
        cells = kwargs.pop("cells", None)
        for key, value in kwargs.items():
            raise TypeError(f"reconstruct() got unexpected keyword argument '{key}'")
        if cells is None:
            cells = self._cells
        return type(self)(cells)

    def __repr__(self):
        """Representation."""
        return f"{type(self).__name__}({self._cells!r})"

    def __str__(self):
        """Format as a string."""
        return f"<{type(self).__name__} #{self._cells}>"

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return ("CellSequence", tuple(c._ufl_hash_data_() for c in self._cells))

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return ("CellSequence", tuple(c._ufl_signature_data_(renumbering) for c in self._cells))

    def _ufl_sort_key_(self):
        """UFL sort key."""
        return ("CellSequence", tuple(c._ufl_sort_key_() for c in self._cells))

    @property
    def cells(self) -> tuple[AbstractCell, ...]:
        """Return the component cells in the CellSequence sense."""
        return self._cells
