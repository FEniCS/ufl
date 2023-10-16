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

from __future__ import annotations

import abc as _abc
import typing as _typing

import numpy as np

from ufl.cell import Cell as _Cell
from ufl.pullback import AbstractPullback as _AbstractPullback
from ufl.pullback import IdentityPullback as _IdentityPullback
from ufl.pullback import MixedPullback as _MixedPullback
from ufl.pullback import SymmetricPullback as _SymmetricPullback
from ufl.sobolevspace import SobolevSpace as _SobolevSpace
from ufl.utils.sequences import product

__all_classes__ = ["AbstractFiniteElement", "FiniteElement", "MixedElement", "SymmetricElement"]


class AbstractFiniteElement(_abc.ABC):
    """Base class for all finite elements.

    To make your element library compatible with UFL, you should make a subclass of AbstractFiniteElement
    and provide implementions of all the abstract methods and properties. All methods and properties
    that are not marked as abstract are implemented here and should not need to be overwritten in your
    subclass.

    An example of how the methods in your subclass could be implemented can be found in Basix; see
    https://github.com/FEniCS/basix/blob/main/python/basix/ufl.py
    """

    @_abc.abstractmethod
    def __repr__(self) -> str:
        """Format as string for evaluation as Python object."""

    @_abc.abstractmethod
    def __str__(self) -> str:
        """Format as string for nice printing."""

    @_abc.abstractmethod
    def __hash__(self) -> int:
        """Return a hash."""

    @_abc.abstractmethod
    def __eq__(self, other: AbstractFiniteElement) -> bool:
        """Check if this element is equal to another element."""

    @_abc.abstractproperty
    def sobolev_space(self) -> _SobolevSpace:
        """Return the underlying Sobolev space."""

    @_abc.abstractproperty
    def pullback(self) -> _AbstractPullback:
        """Return the pullback for this element."""

    @_abc.abstractproperty
    def embedded_superdegree(self) -> _typing.Union[int, None]:
        """Return the degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such that the polynomial
        space of the Lagrange space is a superspace of this element's polynomial space. If this
        element contains basis functions that are not in any Lagrange space, this function should
        return None.

        Note that on a simplex cells, the polynomial space of Lagrange space is a complete polynomial
        space, but on other cells this is not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """

    @_abc.abstractproperty
    def embedded_subdegree(self) -> int:
        """Return the degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space such that the polynomial
        space of the Lagrange space is a subspace of this element's polynomial space. If this
        element's polynomial space does not include the constant function, this function should
        return -1.

        Note that on a simplex cells, the polynomial space of Lagrange space is a complete polynomial
        space, but on other cells this is not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """

    @_abc.abstractproperty
    def cell(self) -> _Cell:
        """Return the cell of the finite element."""

    @_abc.abstractproperty
    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""

    @_abc.abstractproperty
    def sub_elements(self) -> _typing.List:
        """Return list of sub-elements.

        This function does not recurse: ie it does not extract the sub-elements
        of sub-elements.
        """

    def __ne__(self, other: AbstractFiniteElement) -> bool:
        """Check if this element is different to another element."""
        return not self.__eq__(other)

    def is_cellwise_constant(self) -> bool:
        """Check whether this element is spatially constant over each cell."""
        return self.embedded_superdegree == 0

    def _ufl_hash_data_(self) -> str:
        """Return UFL hash data."""
        return repr(self)

    def _ufl_signature_data_(self) -> str:
        """Return UFL signature data."""
        return repr(self)

    @property
    def components(self) -> _typing.Dict[_typing.Tuple[int, ...], int]:
        """Get the numbering of the components of the element.

        Returns:
            A map from the components of the values on a physical cell (eg (0, 1))
            to flat component numbers on the reference cell (eg 1)
        """
        if isinstance(self.pullback, _SymmetricPullback):
            return self.pullback._symmetry

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
        """Return the shape of the value space on the physical domain."""
        return self.pullback.physical_value_shape(self)

    @property
    def value_size(self) -> int:
        """Return the integer product of the value shape."""
        return product(self.value_shape)

    @property
    def reference_value_size(self) -> int:
        """Return the integer product of the reference value shape."""
        return product(self.reference_value_shape)

    @property
    def num_sub_elements(self) -> int:
        """Return number of sub-elements.

        This function does not recurse: ie it does not count the sub-elements of
        sub-elements.
        """
        return len(self.sub_elements)


class FiniteElement(AbstractFiniteElement):
    """A directly defined finite element."""
    __slots__ = ("_repr", "_str", "_family", "_cell", "_degree",
                 "_reference_value_shape", "_pullback", "_sobolev_space",
                 "_sub_elements", "_subdegree")

    def __init__(
        self, family: str, cell: _Cell, degree: int,
        reference_value_shape: _typing.Tuple[int, ...], pullback: _AbstractPullback,
        sobolev_space: _SobolevSpace, sub_elements=[],
        _repr: _typing.Optional[str] = None, _str: _typing.Optional[str] = None,
        subdegree: _typing.Optional[int] = None,
    ):
        """Initialise a finite element.

        This class should only be used for testing

        Args:
            family: The family name of the element
            cell: The cell on which the element is defined
            degree: The polynomial degree of the element
            reference_value_shape: The reference value shape of the element
            pullback: The pullback to use
            sobolev_space: The Sobolev space containing this element
            sub_elements: Sub elements of this element
            _repr: A string representation of this elements
            _str: A string for printing
            subdegree: The embedded subdegree of this element
        """
        if subdegree is None:
            self._subdegree = degree
        else:
            self._subdegree = subdegree
        if _repr is None:
            if len(sub_elements) > 0:
                self._repr = (
                    f"ufl.finiteelement.FiniteElement(\"{family}\", {cell}, {degree}, "
                    f"{reference_value_shape}, {pullback}, {sobolev_space}, {sub_elements!r})")
            else:
                self._repr = (
                    f"ufl.finiteelement.FiniteElement(\"{family}\", {cell}, {degree}, "
                    f"{reference_value_shape}, {pullback}, {sobolev_space})")
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
        self._pullback = pullback
        self._sobolev_space = sobolev_space
        self._sub_elements = sub_elements

    def __repr__(self) -> str:
        """Format as string for evaluation as Python object."""
        return self._repr

    def __str__(self) -> str:
        """Format as string for nice printing."""
        return self._str

    def __hash__(self) -> int:
        """Return a hash."""
        return hash(f"{self!r}")

    def __eq__(self, other) -> bool:
        """Check if this element is equal to another element."""
        return type(self) is type(other) and repr(self) == repr(other)

    @property
    def sobolev_space(self) -> _SobolevSpace:
        """Return the underlying Sobolev space."""
        return self._sobolev_space

    @property
    def pullback(self) -> _AbstractPullback:
        """Return the pullback for this element."""
        return self._pullback

    @property
    def embedded_superdegree(self) -> _typing.Union[int, None]:
        """Return the degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such that the polynomial
        space of the Lagrange space is a superspace of this element's polynomial space. If this
        element contains basis functions that are not in any Lagrange space, this function should
        return None.

        Note that on a simplex cells, the polynomial space of Lagrange space is a complete polynomial
        space, but on other cells this is not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._degree

    @property
    def embedded_subdegree(self) -> int:
        """Return the degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space such that the polynomial
        space of the Lagrange space is a subspace of this element's polynomial space. If this
        element's polynomial space does not include the constant function, this function should
        return -1.

        Note that on a simplex cells, the polynomial space of Lagrange space is a complete polynomial
        space, but on other cells this is not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._subdegree

    @property
    def cell(self) -> _Cell:
        """Return the cell of the finite element."""
        return self._cell

    @property
    def reference_value_shape(self) -> _typing.Tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""
        return self._reference_value_shape

    @property
    def sub_elements(self) -> _typing.List:
        """Return list of sub-elements.

        This function does not recurse: ie it does not extract the sub-elements
        of sub-elements.
        """
        return self._sub_elements


class SymmetricElement(FiniteElement):
    """A symmetric finite element."""

    def __init__(
        self,
        symmetry: _typing.Dict[_typing.Tuple[int, ...], int],
        sub_elements: _typing.List[AbstractFiniteElement]
    ):
        """Initialise a symmetric element.

        This class should only be used for testing

        Args:
            symmetry: Map from physical components to reference components
            sub_elements: Sub-elements of this element
        """
        self._sub_elements = sub_elements
        pullback = _SymmetricPullback(self, symmetry)
        reference_value_shape = (sum(e.reference_value_size for e in sub_elements), )
        degree = max(e.embedded_superdegree for e in sub_elements)
        cell = sub_elements[0].cell
        for e in sub_elements:
            if e.cell != cell:
                raise ValueError("All sub-elements must be defined on the same cell")
        sobolev_space = max(e.sobolev_space for e in sub_elements)

        super().__init__(
            "Symmetric element", cell, degree, reference_value_shape, pullback,
            sobolev_space, sub_elements=sub_elements,
            _repr=(f"ufl.finiteelement.SymmetricElement({symmetry!r}, {sub_elements!r})"),
            _str=f"<symmetric element on a {cell}>")


class MixedElement(FiniteElement):
    """A mixed element."""

    def __init__(self, sub_elements):
        """Initialise a mixed element.

        This class should only be used for testing

        Args:
            sub_elements: Sub-elements of this element
        """
        sub_elements = [MixedElement(e) if isinstance(e, list) else e for e in sub_elements]
        cell = sub_elements[0].cell
        for e in sub_elements:
            assert e.cell == cell
        degree = max(e.embedded_superdegree for e in sub_elements)
        reference_value_shape = (sum(e.reference_value_size for e in sub_elements), )
        if all(isinstance(e.pullback, _IdentityPullback) for e in sub_elements):
            pullback = _IdentityPullback()
        else:
            pullback = _MixedPullback(self)
        sobolev_space = max(e.sobolev_space for e in sub_elements)

        super().__init__(
            "Mixed element", cell, degree, reference_value_shape, pullback, sobolev_space,
            sub_elements=sub_elements,
            _repr=f"ufl.finiteelement.MixedElement({sub_elements!r})",
            _str=f"<MixedElement with {len(sub_elements)} sub-element(s)>")
