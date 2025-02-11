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
from collections.abc import Sequence

from ufl.cell import AbstractCell
from ufl.pullback import AbstractPullback as _AbstractPullback
from ufl.sobolevspace import SobolevSpace as _SobolevSpace
from ufl.utils.sequences import product

__all_classes__ = ["AbstractFiniteElement"]


class AbstractFiniteElement(_abc.ABC):
    """Base class for all finite elements.

    To make your element library compatible with UFL, you should make a
    subclass of AbstractFiniteElement and provide implementions of all
    the abstract methods and properties. All methods and properties that
    are not marked as abstract are implemented here and should not need
    to be overwritten in your subclass.

    An example of how the methods in your subclass could be implemented
    can be found in Basix; see
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
    def __eq__(self, other: object) -> bool:
        """Check if this element is equal to another element."""

    @_abc.abstractproperty
    def sobolev_space(self) -> _SobolevSpace:
        """Return the underlying Sobolev space."""

    @_abc.abstractproperty
    def pullback(self) -> _AbstractPullback:
        """Return the pullback for this element."""

    @_abc.abstractproperty
    def embedded_superdegree(self) -> int | None:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return None.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """

    @_abc.abstractproperty
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """

    @_abc.abstractproperty
    def cell(self) -> AbstractCell:
        """Return the cell of the finite element."""

    @_abc.abstractproperty
    def reference_value_shape(self) -> tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""

    @_abc.abstractproperty
    def sub_elements(self) -> Sequence:
        """Return list of sub-elements.

        This function does not recurse: ie it does not extract the sub-elements
        of sub-elements.
        """

    def __ne__(self, other: object) -> bool:
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
