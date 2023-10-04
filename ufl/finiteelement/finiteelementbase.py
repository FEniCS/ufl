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

from ufl.utils.sequences import product
from ufl.cell import AbstractCell, as_cell
from abc import ABC, abstractmethod


class FiniteElementBase(ABC):
    """Base class for all finite elements."""
    __slots__ = ("_family", "_cell", "_degree", "_quad_scheme",
                 "_value_shape", "_reference_value_shape", "__weakref__")

    # TODO: Not all these should be in the base class! In particular
    # family, degree, and quad_scheme do not belong here.
    def __init__(self, family, cell, degree, quad_scheme, value_shape,
                 reference_value_shape):
        """Initialize basic finite element data."""
        if not (degree is None or isinstance(degree, (int, tuple))):
            raise ValueError("Invalid degree type.")
        if not isinstance(value_shape, tuple):
            raise ValueError("Invalid value_shape type.")
        if not isinstance(reference_value_shape, tuple):
            raise ValueError("Invalid reference_value_shape type.")

        if cell is not None:
            cell = as_cell(cell)
            if not isinstance(cell, AbstractCell):
                raise ValueError("Invalid cell type.")

        self._family = family
        self._cell = cell
        self._degree = degree
        self._value_shape = value_shape
        self._reference_value_shape = reference_value_shape
        self._quad_scheme = quad_scheme

    @abstractmethod
    def __repr__(self):
        """Format as string for evaluation as Python object."""
        pass

    @abstractmethod
    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        pass

    @abstractmethod
    def mapping(self):
        """Return the mapping type for this element."""
        pass

    def _is_globally_constant(self):
        """Check if the element is a global constant.

        For Real elements, this should return True.
        """
        return False

    def _is_linear(self):
        """Check if the element is Lagrange degree 1."""
        return False

    def _ufl_hash_data_(self):
        """Doc."""
        return repr(self)

    def _ufl_signature_data_(self):
        """Doc."""
        return repr(self)

    def __hash__(self):
        """Compute hash code for insertion in hashmaps."""
        return hash(self._ufl_hash_data_())

    def __eq__(self, other):
        """Compute element equality for insertion in hashmaps."""
        return type(self) is type(other) and self._ufl_hash_data_() == other._ufl_hash_data_()

    def __ne__(self, other):
        """Compute element inequality for insertion in hashmaps."""
        return not self.__eq__(other)

    def __lt__(self, other):
        """Compare elements by repr, to give a natural stable sorting."""
        return repr(self) < repr(other)

    def family(self):  # FIXME: Undefined for base?
        """Return finite element family."""
        return self._family

    def variant(self):
        """Return the variant used to initialise the element."""
        return None

    def degree(self, component=None):
        """Return polynomial degree of finite element."""
        # FIXME: Consider embedded_degree concept for more accurate
        # degree, see blueprint
        return self._degree

    def quadrature_scheme(self):
        """Return quadrature scheme of finite element."""
        return self._quad_scheme

    def cell(self):
        """Return cell of finite element."""
        return self._cell

    def is_cellwise_constant(self, component=None):
        """Return whether the basis functions of this element is spatially constant over each cell."""
        return self._is_globally_constant() or self.degree() == 0

    def value_shape(self):
        """Return the shape of the value space on the global domain."""
        return self._value_shape

    def reference_value_shape(self):
        """Return the shape of the value space on the reference cell."""
        return self._reference_value_shape

    def value_size(self):
        """Return the integer product of the value shape."""
        return product(self.value_shape())

    def reference_value_size(self):
        """Return the integer product of the reference value shape."""
        return product(self.reference_value_shape())

    def symmetry(self):  # FIXME: different approach
        r"""Return the symmetry dict.

        This is a mapping :math:`c_0 \\to c_1`
        meaning that component :math:`c_0` is represented by component
        :math:`c_1`.
        A component is a tuple of one or more ints.
        """
        return {}

    def _check_component(self, i):
        """Check that component index i is valid."""
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j, k) in zip(i, sh))):
            raise ValueError(
                f"Illegal component index {i} (value rank {len(i)}) "
                f"for element (value rank {r}).")

    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative component index for a given component index."""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (None, i)

    def extract_component(self, i):
        """Recursively extract component index relative to a (simple) element.

        and that element for given value component index.
        """
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (i, self)

    def _check_reference_component(self, i):
        """Check that reference component index i is valid."""
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j, k) in zip(i, sh))):
            raise ValueError(
                f"Illegal component index {i} (value rank {len(i)}) "
                f"for element (value rank {r}).")

    def extract_subelement_reference_component(self, i):
        """Extract direct subelement index and subelement relative.

        reference component index for a given reference component index.
        """
        if isinstance(i, int):
            i = (i,)
        self._check_reference_component(i)
        return (None, i)

    def extract_reference_component(self, i):
        """Recursively extract reference component index relative to a (simple) element.

        and that element for given reference value component index.
        """
        if isinstance(i, int):
            i = (i,)
        self._check_reference_component(i)
        return (i, self)

    def num_sub_elements(self):
        """Return number of sub-elements."""
        return 0

    def sub_elements(self):
        """Return list of sub-elements."""
        return []

    def __add__(self, other):
        """Add two elements, creating an enriched element."""
        if not isinstance(other, FiniteElementBase):
            raise ValueError(f"Can't add element and {other.__class__}.")
        from ufl.finiteelement import EnrichedElement
        return EnrichedElement(self, other)

    def __mul__(self, other):
        """Multiply two elements, creating a mixed element."""
        if not isinstance(other, FiniteElementBase):
            raise ValueError("Can't multiply element and {other.__class__}.")
        from ufl.finiteelement import MixedElement
        return MixedElement(self, other)

    def __getitem__(self, index):
        """Restrict finite element to a subdomain, subcomponent or topology (cell)."""
        if index in ("facet", "interior"):
            from ufl.finiteelement import RestrictedElement
            return RestrictedElement(self, index)
        else:
            raise KeyError(f"Invalid index for restriction: {repr(index)}")

    def __iter__(self):
        """Iter."""
        raise TypeError(f"'{type(self).__name__}' object is not iterable")
