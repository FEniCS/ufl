# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

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

from ufl.sobolevspace import SobolevSpace as _SobolevSpace
from ufl.utils.indexflattening import shape_to_strides, unflatten_index
from ufl.utils.sequences import product

__all_classes__ = ["AbstractFiniteElement", "FiniteElement", "MixedElement"]


class AbstractFiniteElement(_abc.ABC):
    """Base class for all finite elements.

    TODO: instructions for making subclasses of this.
    """

    @_abc.abstractmethod
    def __repr__(self) -> str:
        """Format as string for evaluation as Python object.

        This representation must include all the input arguments of the element
        as it will be used to check for element equality.
        """

    @_abc.abstractmethod
    def __str__(self) -> str:
        """Format as string for nice printing."""

    @_abc.abstractproperty
    def sobolev_space(self) -> _SobolevSpace:
        """Return the underlying Sobolev space."""

    @_abc.abstractproperty
    def mapping(self) -> str:
        """Return the mapping type for this element."""

    @_abc.abstractproperty
    def embedded_degree(self) -> int:
        """The maximum degree of a polynomial included in the basis for this element."""

    @_abc.abstractproperty
    def cell(self) -> str:
        """Return the cell type of the finite element."""

    @_abc.abstractproperty
    def value_shape(self) -> _typing.Tuple[int]:
        """Return the shape of the value space on the global domain."""

    @_abc.abstractproperty
    def reference_value_shape(self) -> _typing.Tuple[int]:
        """Return the shape of the value space on the reference cell."""

    @_abc.abstractproperty
    def _is_globally_constant(self) -> bool:
        """Check if the element is a global constant.

        For Real elements, this should return True.
        """

    @_abc.abstractproperty
    def _is_cellwise_constant(self):
        """Return whether the basis functions of this element are constant over each cell."""

    @_abc.abstractproperty
    def _is_linear(self) -> bool:
        """Check if the element is Lagrange degree 1."""

    @property
    def value_size(self) -> int:
        """Return the integer product of the value shape."""
        return product(self.value_shape)

    @property
    def reference_value_size(self) -> int:
        """Return the integer product of the reference value shape."""
        return product(self.reference_value_shape)

    @_abc.abstractproperty
    def sub_elements(self):
        """Return list of sub-elements."""

    @property
    def num_sub_elements(self):
        """Return number of sub-elements."""
        return len(self.sub_elements)

    # Stuff below here needs thinking about
    def _ufl_hash_data_(self):
        return repr(self)

    def _ufl_signature_data_(self):
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

    def symmetry(self):  # FIXME: different approach
        """Return the symmetry dict, which is a mapping :math:`c_0 \\to c_1`
        meaning that component :math:`c_0` is represented by component
        :math:`c_1`.
        A component is a tuple of one or more ints."""
        return {}

    def _check_component(self, i):
        """Check that component index i is valid"""
        sh = self.value_shape
        r = len(sh)
        if not (len(i) == r and all(j < k for (j, k) in zip(i, sh))):
            raise ValueError(
                f"Illegal component index {i} (value rank {len(i)}) "
                f"for element (value rank {r}).")

    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative
        component index for a given component index."""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (None, i)

    def extract_component(self, i):
        """Recursively extract component index relative to a (simple) element
        and that element for given value component index."""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (i, self)

    def _check_reference_component(self, i):
        "Check that reference component index i is valid."
        sh = self.value_shape
        r = len(sh)
        if not (len(i) == r and all(j < k for (j, k) in zip(i, sh))):
            raise ValueError(
                f"Illegal component index {i} (value rank {len(i)}) "
                f"for element (value rank {r}).")

    def extract_subelement_reference_component(self, i):
        """Extract direct subelement index and subelement relative
        reference component index for a given reference component index."""
        if isinstance(i, int):
            i = (i,)
        self._check_reference_component(i)
        return (None, i)

    def extract_reference_component(self, i):
        """Recursively extract reference component index relative to a (simple) element
        and that element for given reference value component index."""
        if isinstance(i, int):
            i = (i,)
        self._check_reference_component(i)
        return (i, self)

    @property
    def flattened_sub_element_mapping(self):
        return None


class FiniteElement(AbstractFiniteElement):
    """A directly defined finite element."""
    __slots__ = ("_repr", "_str", "_family", "_cell", "_degree", "_value_shape",
                 "_reference_value_shape", "_mapping", "_sobolev_space", "_component_map",
                 "_sub_elements")

    def __init__(self, family, cell, degree, value_shape,
                 reference_value_shape, mapping, sobolev_space, component_map=None, sub_elements=[]):
        """Initialize basic finite element data."""
        if component_map is None:
            self._repr = (f"ufl.finiteelement.FiniteElement(\"{family}\", {cell}, {degree}, {value_shape}, "
                          f"{reference_value_shape}, \"{mapping}\", {sobolev_space})")
        else:
            self._repr = (f"ufl.finiteelement.FiniteElement(\"{family}\", {cell}, {degree}, {value_shape}, "
                          f"{reference_value_shape}, \"{mapping}\", {sobolev_space}, component_map={component_map})")
        self._str = f"<{family}{degree} on a {cell}>"
        self._family = family
        self._cell = cell
        self._degree = degree
        self._value_shape = value_shape
        self._reference_value_shape = reference_value_shape
        self._mapping = mapping
        self._sobolev_space = sobolev_space
        self._component_map = component_map
        self._sub_elements = sub_elements

    def __repr__(self):
        """Format as string for evaluation as Python object."""
        return self._repr

    def __str__(self) -> str:
        """Format as string for nice printing."""
        return self._str

    @property
    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return self._sobolev_space

    @property
    def mapping(self):
        """Return the mapping type for this element."""
        return self._mapping

    @property
    def embedded_degree(self) -> int:
        """The maximum degree of a polynomial included in the basis for this element."""
        return self._degree

    @property
    def cell(self) -> str:
        """Return the cell type of the finite element."""
        return self._cell

    @property
    def value_shape(self) -> _typing.Tuple[int]:
        """Return the shape of the value space on the global domain."""
        return self._value_shape

    @property
    def reference_value_shape(self) -> _typing.Tuple[int]:
        """Return the shape of the value space on the reference cell."""
        return self._reference_value_shape

    @property
    def _is_globally_constant(self) -> bool:
        """Check if the element is a global constant.

        For Real elements, this should return True.
        """
        return self._family == "Real"

    @property
    def _is_cellwise_constant(self):
        """Return whether the basis functions of this element are constant over each cell."""
        return self._is_globally_constant or self._degree == 0

    @property
    def _is_linear(self) -> bool:
        """Check if the element is Lagrange degree 1."""
        return self._family == "Lagrange" and self._degree == 1

    @property
    def sub_elements(self):
        """Return list of sub-elements."""
        return self._sub_elements

    # FIXME: functions below this comment are hacks
    def symmetry(self):
        if self._component_map is None:
            return {}
        s = {}
        out = {}
        for i, j in self._component_map.items():
            if j in s:
                out[i] = s[j]
            else:
                s[j] = i
        return out

    @property
    def flattened_sub_element_mapping(self):
        if self._component_map is None:
            return None
        else:
            return list(self._component_map.values())


class MixedElement(AbstractFiniteElement):
    """A mixed element."""
    __slots__ = ["_repr", "_str", "_subelements", "_cell"]

    def __init__(self, subelements):
        self._repr = f"ufl.finiteelement.MixedElement({subelements!r})"
        self._str = f"<MixedElement with {len(subelements)} subelement(s)>"
        self._subelements = [MixedElement(e) if isinstance(e, list) else e for e in subelements]
        self._cell = self._subelements[0].cell
        for e in self._subelements:
            assert e.cell == self._cell

    def __repr__(self):
        """Format as string for evaluation as Python object."""
        return self._repr

    def __str__(self) -> str:
        """Format as string for nice printing."""
        return self._str

    @property
    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return max(e.sobolev_space for e in self._subelements)

    @property
    def mapping(self):
        """Return the mapping type for this element."""
        if all(e.mapping == "identity" for e in self._subelements):
            return "identity"
        else:
            return "undefined"

    @property
    def embedded_degree(self) -> int:
        """The maximum degree of a polynomial included in the basis for this element."""
        return max(e.embedded_degree for e in self._subelements)

    @property
    def cell(self) -> str:
        """Return the cell type of the finite element."""
        return self._cell

    @property
    def value_shape(self) -> _typing.Tuple[int]:
        """Return the shape of the value space on the global domain."""
        return (sum(e.value_size for e in self._subelements), )

    @property
    def reference_value_shape(self) -> _typing.Tuple[int]:
        """Return the shape of the value space on the reference cell."""
        return (sum(e.reference_value_size for e in self._subelements), )

    @property
    def _is_globally_constant(self) -> bool:
        """Check if the element is a global constant.

        For Real elements, this should return True.
        """
        return all(e._is_globally_constant for e in self._subelements)

    @property
    def _is_cellwise_constant(self):
        """Return whether the basis functions of this element are constant over each cell."""
        return all(e._is_cellwise_constant for e in self._subelements)

    @property
    def _is_linear(self) -> bool:
        """Check if the element is Lagrange degree 1."""
        return all(e._is_linear for e in self._subelements)

    @property
    def sub_elements(self):
        """Return list of sub-elements."""
        return self._subelements

    # FIXME: functions below this comment are hacks
    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative
        component index for a given component index."""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)

        # Select between indexing modes
        if len(self.value_shape) == 1:
            # Indexing into a long vector of flattened subelement
            # shapes
            j, = i

            # Find subelement for this index
            for sub_element_index, e in enumerate(self.sub_elements):
                sh = e.value_shape
                si = product(sh)
                if j < si:
                    break
                j -= si
            if j < 0:
                raise ValueError("Moved past last value component!")

            # Convert index into a shape tuple
            st = shape_to_strides(sh)
            component = unflatten_index(j, st)
        else:
            # Indexing into a multidimensional tensor where subelement
            # index is first axis
            sub_element_index = i[0]
            if sub_element_index >= len(self.sub_elements):
                raise ValueError(f"Illegal component index (dimension {sub_element_index}).")
            component = i[1:]
        return (sub_element_index, component)