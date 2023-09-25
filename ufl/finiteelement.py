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

import numpy as np

from ufl.cell import Cell as _Cell
from ufl.pull_back import AbstractPullBack as _AbstractPullBack
from ufl.pull_back import IdentityPullBack as _IdentityPullBack
from ufl.pull_back import MixedPullBack as _MixedPullBack
from ufl.pull_back import SymmetricPullBack as _SymmetricPullBack
from ufl.sobolevspace import SobolevSpace as _SobolevSpace
from ufl.utils.sequences import product

__all_classes__ = ["AbstractFiniteElement", "FiniteElement", "MixedElement", "SymmetricElement"]


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
    def embedded_superdegree(self) -> _typing.Union[int, None]:
        """The maximum degree of a polynomial included in the basis for this element.

        This returns the degree of the lowest degree Lagrange space such that the polynomial
        space of the Lagrange space is a superset of this element's polynomial space. If this
        element contains basis functions that are not in any Lagrange space, this function should
        return None.
        """

    @_abc.abstractproperty
    def embedded_subdegree(self) -> int:
        """The maximum degree Lagrange space that is a subset of this element.

        This returns the degree of the highest degree Lagrange space such that the polynomial
        space of the Lagrange space is a subset of this element's polynomial space. If this
        element's polynomial space does not included the constant function, this function should
        return -1.
        """

    @_abc.abstractproperty
    def cell(self) -> _Cell:
        """Return the cell type of the finite element."""

    @_abc.abstractproperty
    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""

    @property
    def components(self) -> _typing.Dict[_typing.Tuple[int, ...], int]:
        """Get the numbering of the components of the element."""
        if isinstance(self.pull_back, _SymmetricPullBack):
            return self.pull_back._symmetry

        if len(self.sub_elements) == 0:
            return {(): 0}

        components = {}
        offset = 0
        c_offset = 0
        for e in self.sub_elements:
            for i, j in enumerate(np.ndindex(e.value_shape)):
                components[(offset + i, )] = c_offset + e.components[j]
            c_offset += max(e.components.values()) + 1
            offset += e.value_size
        return components

    @property
    def value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the global domain."""
        return self.pull_back.physical_value_shape(self)

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
        return self.embedded_superdegree == 0

    @_abc.abstractmethod
    def __hash__(self) -> int:
        """Compute hash code."""

    @_abc.abstractmethod
    def __eq__(self, other) -> bool:
        """Check element equality."""

    def __ne__(self, other) -> bool:
        """Check element inequality."""
        return not self.__eq__(other)

    def _ufl_hash_data_(self) -> str:
        """Return UFL hash data."""
        return repr(self)

    def _ufl_signature_data_(self) -> str:
        """Return UFL signature data."""
        return repr(self)


class FiniteElement(AbstractFiniteElement):
    """A directly defined finite element."""
    __slots__ = ("_repr", "_str", "_family", "_cell", "_degree",
                 "_reference_value_shape", "_pull_back", "_sobolev_space",
                 "_sub_elements", "_subdegree")

    def __init__(
        self, family: str, cell: _Cell, degree: int,
        reference_value_shape: _typing.Tuple[int, ...], pull_back: _AbstractPullBack,
        sobolev_space: _SobolevSpace, sub_elements=[],
        _repr: _typing.Optional[str] = None, _str: _typing.Optional[str] = None,
        subdegree: _typing.Optional[int] = None,
    ):
        """Initialize basic finite element data."""
        if subdegree is None:
            self._subdegree = degree
        else:
            self._subdegree = subdegree
        if _repr is None:
            if len(sub_elements) > 0:
                self._repr = (
                    f"ufl.finiteelement.FiniteElement(\"{family}\", {cell}, {degree}, "
                    f"{reference_value_shape}, {pull_back}, {sobolev_space}, {sub_elements!r})")
            else:
                self._repr = (
                    f"ufl.finiteelement.FiniteElement(\"{family}\", {cell}, {degree}, "
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
    def embedded_superdegree(self) -> _typing.Union[int, None]:
        """The maximum degree of a polynomial included in the basis for this element."""
        return self._degree

    @property
    def embedded_subdegree(self) -> int:
        """The maximum degree Lagrange space that is a subset of this element."""
        return self._subdegree

    @property
    def cell(self) -> _Cell:
        """Return the cell type of the finite element."""
        return self._cell

    @property
    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""
        return self._reference_value_shape

    @property
    def sub_elements(self) -> _typing.List:
        """Return list of sub-elements."""
        return self._sub_elements

    def __hash__(self) -> int:
        """Compute hash code."""
        return hash(f"{self!r}")

    def __eq__(self, other) -> bool:
        """Check element equality."""
        return type(self) is type(other) and repr(self) == repr(other)


class SymmetricElement(FiniteElement):
    """A symmetric finite element."""

    def __init__(
        self,
        symmetry: _typing.Dict[_typing.Tuple[int, ...], int],
        sub_elements: _typing.List[AbstractFiniteElement]
    ):
        """Initialise."""
        pull_back = _SymmetricPullBack(self, symmetry)
        reference_value_shape = (sum(e.reference_value_size for e in sub_elements), )
        degree = max(e.embedded_superdegree for e in sub_elements)
        cell = sub_elements[0].cell
        for e in sub_elements:
            if e.cell != cell:
                raise ValueError("All sub-elements must be defined on the same cell")
        sobolev_space = max(e.sobolev_space for e in sub_elements)

        super().__init__(
            "Symmetric element", cell, degree, reference_value_shape, pull_back,
            sobolev_space, sub_elements=sub_elements,
            _repr=(f"ufl.finiteelement.SymmetricElement({symmetry!r}, {sub_elements!r})"),
            _str=f"<symmetric element on a {cell}>")


class MixedElement(FiniteElement):
    """A mixed element."""

    def __init__(self, sub_elements):
        """Initialise."""
        sub_elements = [MixedElement(e) if isinstance(e, list) else e for e in sub_elements]
        cell = sub_elements[0].cell
        for e in sub_elements:
            assert e.cell == cell
        degree = max(e.embedded_superdegree for e in sub_elements)
        reference_value_shape = (sum(e.reference_value_size for e in sub_elements), )
        if all(isinstance(e.pull_back, _IdentityPullBack) for e in sub_elements):
            pull_back = _IdentityPullBack()
        else:
            pull_back = _MixedPullBack(self)
        sobolev_space = max(e.sobolev_space for e in sub_elements)

        super().__init__(
            "Mixed element", cell, degree, reference_value_shape, pull_back, sobolev_space,
            sub_elements=sub_elements,
            _repr=f"ufl.finiteelement.MixedElement({sub_elements!r})",
            _str=f"<MixedElement with {len(sub_elements)} sub-element(s)>")
