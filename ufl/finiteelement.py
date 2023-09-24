"""This module defines the UFL finite element classes."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Massimiliano Leoni, 2016
# Modified by Matthew Scroggs, 2023

import abc as _abc
import typing as _typing

from ufl.cell import Cell as _Cell
from ufl.pull_back import AbstractPullBack as _AbstractPullBack
from ufl.pull_back import IdentityPullBack as _IdentityPullBack
from ufl.pull_back import MixedPullBack as _MixedPullBack
from ufl.pull_back import SymmetricPullBack as _SymmetricPullBack
from ufl.sobolevspace import SobolevSpace as _SobolevSpace
from ufl.utils.sequences import product

__all_classes__ = ["AbstractFiniteElement", "FiniteElement", "MixedElement"]


class AbstractFiniteElement(_abc.ABC):
    """Base class for all finite elements.

    TODO: instructions for making subclasses of this.
    """

    @_abc.abstractmethod
    def __repr__(self) -> str:
        """Format as string for evaluation as Python object."""

    @_abc.abstractmethod
    def __str__(self) -> str:
        """Format as string for nice printing."""

    @_abc.abstractproperty
    def sobolev_space(self) -> _SobolevSpace:
        """Return the underlying Sobolev space."""

    @_abc.abstractproperty
    def pull_back(self) -> _AbstractPullBack:
        """Return the pull back map for this element."""

    @_abc.abstractproperty
    def embedded_degree(self) -> int:
        """The maximum degree of a polynomial included in the basis for this element."""

    @_abc.abstractproperty
    def cell(self) -> _Cell:
        """Return the cell type of the finite element."""

    @_abc.abstractproperty
    def value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the global domain."""

    @_abc.abstractproperty
    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""

    @property
    def value_size(self) -> int:
        """Return the integer product of the value shape."""
        return product(self.value_shape)

    @property
    def reference_value_size(self) -> int:
        """Return the integer product of the reference value shape."""
        return product(self.reference_value_shape)

    @_abc.abstractproperty
    def sub_elements(self) -> _typing.List:
        """Return list of sub-elements.

        This function does not recurse: ie it does not extract the sub-elements
        of sub-elements.
        """

    @property
    def num_sub_elements(self) -> int:
        """Return number of sub-elements.

        This function does not recurse: ie it does not count the sub-elements of
        sub-elements.
        """
        return len(self.sub_elements)

    def is_cellwise_constant(self) -> bool:
        """Return whether this element is spatially constant over each cell."""
        return self.embedded_degree == 0

    # Stuff below here needs thinking about
    def _ufl_hash_data_(self) -> str:
        return repr(self)

    def _ufl_signature_data_(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        """Compute hash code for insertion in hashmaps."""
        return hash(self._ufl_hash_data_())

    def __eq__(self, other) -> bool:
        """Compute element equality for insertion in hashmaps."""
        return type(self) is type(other) and self._ufl_hash_data_() == other._ufl_hash_data_()

    def __ne__(self, other) -> bool:
        """Compute element inequality for insertion in hashmaps."""
        return not self.__eq__(other)


class FiniteElement(AbstractFiniteElement):
    """A directly defined finite element."""
    __slots__ = ("_repr", "_str", "_family", "_cell", "_degree", "_value_shape",
                 "_reference_value_shape", "_pull_back", "_sobolev_space",
                 "_sub_elements")

    def __init__(
        self, family: str, cell: _Cell, degree: int, value_shape: _typing.Tuple[int, ...],
        reference_value_shape: _typing.Tuple[int, ...], pull_back: _AbstractPullBack,
        sobolev_space: _SobolevSpace, sub_elements=[],
        _repr: _typing.Optional[str] = None, _str: _typing.Optional[str] = None
    ):
        """Initialize basic finite element data."""
        if _repr is None:
            if len(sub_elements) > 0:
                self._repr = (
                    f"ufl.finiteelement.FiniteElement(\"{family}\", {cell}, {degree}, {value_shape}, "
                    f"{reference_value_shape}, {pull_back}, {sobolev_space}, {sub_elements!r})")
            else:
                self._repr = (
                    f"ufl.finiteelement.FiniteElement(\"{family}\", {cell}, {degree}, {value_shape}, "
                    f"{reference_value_shape}, {pull_back}, {sobolev_space})")
        else:
            self._repr = _repr
        if _str is None:
            self._str = f"<{family}{degree} on a {cell}>"
        else:
            self._str = _str
        self._family = family
        self._cell = cell
        self._degree = degree
        self._value_shape = value_shape
        self._reference_value_shape = reference_value_shape
        self._pull_back = pull_back
        self._sobolev_space = sobolev_space
        self._sub_elements = sub_elements

    def __repr__(self) -> str:
        """Format as string for evaluation as Python object."""
        return self._repr

    def __str__(self) -> str:
        """Format as string for nice printing."""
        return self._str

    @property
    def sobolev_space(self) -> _SobolevSpace:
        """Return the underlying Sobolev space."""
        return self._sobolev_space

    @property
    def pull_back(self) -> _AbstractPullBack:
        """Return the pull back map for this element."""
        return self._pull_back

    @property
    def embedded_degree(self) -> int:
        """The maximum degree of a polynomial included in the basis for this element."""
        return self._degree

    @property
    def cell(self) -> _Cell:
        """Return the cell type of the finite element."""
        return self._cell

    @property
    def value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the global domain."""
        return self._value_shape

    @property
    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""
        return self._reference_value_shape

    @property
    def sub_elements(self) -> _typing.List:
        """Return list of sub-elements."""
        return self._sub_elements


class SymmetricElement(FiniteElement):
    """A symmetric finite element."""

    def __init__(
        self, value_shape: _typing.Tuple[int, ...],
        symmetry: _typing.Dict[_typing.Tuple[int, ...], int],
        sub_elements: _typing.List[AbstractFiniteElement]
    ):
        """Initialise."""
        pull_back = _SymmetricPullBack(self, symmetry)
        reference_value_shape = (sum(e.reference_value_size for e in sub_elements), )
        degree = max(e.embedded_degree for e in sub_elements)
        cell = sub_elements[0].cell
        for e in sub_elements:
            if e.cell != cell:
                raise ValueError("All sub-elements must be defined on the same cell")
        sobolev_space = max(e.sobolev_space for e in sub_elements)

        super().__init__(
            "Symmetric element", cell, degree, value_shape, reference_value_shape, pull_back,
            sobolev_space, sub_elements=sub_elements,
            _repr=(f"ufl.finiteelement.SymmetricElement({value_shape}, {symmetry!r}, {sub_elements!r})"),
            _str=f"<symmetric element on a {cell}>")


class MixedElement(AbstractFiniteElement):
    """A mixed element."""
    __slots__ = ["_repr", "_str", "_subelements", "_cell"]

    def __init__(self, subelements):
        """Initialise."""
        self._repr = f"ufl.finiteelement.MixedElement({subelements!r})"
        self._str = f"<MixedElement with {len(subelements)} subelement(s)>"
        self._subelements = [MixedElement(e) if isinstance(e, list) else e for e in subelements]
        self._cell = self._subelements[0].cell
        for e in self._subelements:
            assert e.cell == self._cell

    def __repr__(self) -> str:
        """Format as string for evaluation as Python object."""
        return self._repr

    def __str__(self) -> str:
        """Format as string for nice printing."""
        return self._str

    @property
    def sobolev_space(self) -> _SobolevSpace:
        """Return the underlying Sobolev space."""
        return max(e.sobolev_space for e in self._subelements)

    @property
    def pull_back(self) -> _AbstractPullBack:
        """Return the pull back map for this element."""
        if all(isinstance(e.pull_back, _IdentityPullBack) for e in self._subelements):
            return _IdentityPullBack()
        else:
            return _MixedPullBack(self)

    @property
    def embedded_degree(self) -> int:
        """The maximum degree of a polynomial included in the basis for this element."""
        return max(e.embedded_degree for e in self._subelements)

    @property
    def cell(self) -> _Cell:
        """Return the cell type of the finite element."""
        return self._cell

    @property
    def value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the global domain."""
        return (sum(e.value_size for e in self._subelements), )

    @property
    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""
        return (sum(e.reference_value_size for e in self._subelements), )

    @property
    def sub_elements(self) -> _typing.List:
        """Return list of sub-elements."""
        return self._subelements
