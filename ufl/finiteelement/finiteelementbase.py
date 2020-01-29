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

from ufl.utils.sequences import product
from ufl.utils.dicts import EmptyDict
from ufl.log import error
from ufl.cell import AbstractCell, as_cell


class FiniteElementBase(object):
    "Base class for all finite elements."
    __slots__ = ("_family",
                 "_cell",
                 "_degree",
                 "_quad_scheme",
                 "_value_shape",
                 "_reference_value_shape",
                 "_repr",
                 "__weakref__")

    # TODO: Not all these should be in the base class! In particular
    # family, degree, and quad_scheme do not belong here.
    def __init__(self, family, cell, degree, quad_scheme, value_shape,
                 reference_value_shape):
        "Initialize basic finite element data."
        if not isinstance(family, str):
            error("Invalid family type.")
        if not (degree is None or isinstance(degree, (int, tuple))):
            error("Invalid degree type.")
        if not isinstance(value_shape, tuple):
            error("Invalid value_shape type.")
        if not isinstance(reference_value_shape, tuple):
            error("Invalid reference_value_shape type.")

        if cell is not None:
            cell = as_cell(cell)
            if not isinstance(cell, AbstractCell):
                error("Invalid cell type.")

        self._family = family
        self._cell = cell
        self._degree = degree
        self._value_shape = value_shape
        self._reference_value_shape = reference_value_shape
        self._quad_scheme = quad_scheme

    def __repr__(self):
        """Format as string for evaluation as Python object.

        NB! Assumes subclass has assigned its repr string to self._repr.
        """
        return self._repr

    def _ufl_hash_data_(self):
        return repr(self)

    def _ufl_signature_data_(self):
        return repr(self)

    def __hash__(self):
        "Compute hash code for insertion in hashmaps."
        return hash(self._ufl_hash_data_())

    def __eq__(self, other):
        "Compute element equality for insertion in hashmaps."
        return type(self) == type(other) and self._ufl_hash_data_() == other._ufl_hash_data_()

    def __ne__(self, other):
        "Compute element inequality for insertion in hashmaps."
        return not self.__eq__(other)

    def __lt__(self, other):
        "Compare elements by repr, to give a natural stable sorting."
        return repr(self) < repr(other)

    def family(self):  # FIXME: Undefined for base?
        "Return finite element family."
        return self._family

    def degree(self, component=None):
        "Return polynomial degree of finite element."
        # FIXME: Consider embedded_degree concept for more accurate
        # degree, see blueprint
        return self._degree

    def quadrature_scheme(self):
        "Return quadrature scheme of finite element."
        return self._quad_scheme

    def mapping(self):
        "Not implemented."
        error("Missing implementation of mapping().")

    def cell(self):
        "Return cell of finite element."
        return self._cell

    def is_cellwise_constant(self, component=None):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        return self.family() == "Real" or self.degree() == 0

    def value_shape(self):
        "Return the shape of the value space on the global domain."
        return self._value_shape

    def reference_value_shape(self):
        "Return the shape of the value space on the reference cell."
        return self._reference_value_shape

    def value_size(self):
        "Return the integer product of the value shape."
        return product(self.value_shape())

    def reference_value_size(self):
        "Return the integer product of the reference value shape."
        return product(self.reference_value_shape())

    def symmetry(self):  # FIXME: different approach
        """Return the symmetry dict, which is a mapping :math:`c_0 \\to c_1`
        meaning that component :math:`c_0` is represented by component
        :math:`c_1`.
        A component is a tuple of one or more ints."""
        return EmptyDict

    def _check_component(self, i):
        "Check that component index i is valid"
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j, k) in zip(i, sh))):
            error(("Illegal component index '%s' (value rank %d)" +
                   "for element (value rank %d).") % (i, len(i), r))

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
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j, k) in zip(i, sh))):
            error(("Illegal component index '%s' (value rank %d)" +
                   "for element (value rank %d).") % (i, len(i), r))

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

    def num_sub_elements(self):
        "Return number of sub-elements."
        return 0

    def sub_elements(self):
        "Return list of sub-elements."
        return []

    def __add__(self, other):
        "Add two elements, creating an enriched element"
        if not isinstance(other, FiniteElementBase):
            error("Can't add element and %s." % other.__class__)
        from ufl.finiteelement import EnrichedElement
        return EnrichedElement(self, other)

    def __mul__(self, other):
        "Multiply two elements, creating a mixed element"
        if not isinstance(other, FiniteElementBase):
            error("Can't multiply element and %s." % other.__class__)
        from ufl.finiteelement import MixedElement
        return MixedElement(self, other)

    def __getitem__(self, index):
        "Restrict finite element to a subdomain, subcomponent or topology (cell)."
        if index in ("facet", "interior"):
            from ufl.finiteelement import RestrictedElement
            return RestrictedElement(self, index)
        return NotImplemented
