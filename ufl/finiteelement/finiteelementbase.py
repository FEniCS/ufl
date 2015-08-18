# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

# Copyright (C) 2008-2014 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012

from six.moves import zip
from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.common import product, istr, EmptyDict
from ufl.geometry import Cell, as_cell, as_domain, Domain
from ufl.log import info_blue, warning, warning_blue, error


class FiniteElementBase(object):
    "Base class for all finite elements"
    __slots__ = ("_family",
                 "_cell", "_domain",
                 "_degree",
                 "_form_degree",
                 "_quad_scheme",
                 "_value_shape",
                 "_reference_value_shape",
                 "_repr",
                 "__weakref__")

    def __init__(self, family, domain, degree, quad_scheme, value_shape, reference_value_shape):
        "Initialize basic finite element data"
        ufl_assert(isinstance(family, str), "Invalid family type.")
        ufl_assert(isinstance(degree, (int, tuple)) or degree is None, "Invalid degree type.")
        ufl_assert(isinstance(value_shape, tuple), "Invalid value_shape type.")
        ufl_assert(isinstance(reference_value_shape, tuple), "Invalid reference_value_shape type.")

        # TODO: Support multiple domains for composite mesh mixed elements
        if domain is None:
            self._domain = None
            self._cell = None
        else:
            self._domain = as_domain(domain)
            self._cell = self._domain.cell()
            ufl_assert(isinstance(self._domain, Domain), "Invalid domain type.")
            ufl_assert(isinstance(self._cell, Cell), "Invalid cell type.")

        self._family = family
        self._degree = degree
        self._value_shape = value_shape
        self._reference_value_shape = reference_value_shape
        self._quad_scheme = quad_scheme

    def __repr__(self):
        "Format as string for evaluation as Python object."
        return self._repr

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data, which must be reconstructed or supplied by other means.
        """
        raise NotImplementedError("Class %s must implement FiniteElementBase.reconstruction_signature" % (type(self).__name__,))

    def signature_data(self, renumbering):
        data = ("FiniteElementBase", self._family, self._degree,
                self._value_shape, self._reference_value_shape,
                self._quad_scheme,
                ("no domain" if self._domain is None else self._domain.signature_data(renumbering)))
        return data

    def __hash__(self):
        "Compute hash code for insertion in hashmaps."
        return hash(repr(self))

    def __eq__(self, other):
        "Compute element equality for insertion in hashmaps."
        return type(self) == type(other) and repr(self) == repr(other)

    def __lt__(self, other):
        "Compare elements by repr, to give a natural stable sorting."
        return repr(self) < repr(other)

    def family(self): # FIXME: Undefined for base?
        "Return finite element family"
        return self._family

    def degree(self, component=None):
        "Return polynomial degree of finite element"
        # FIXME: Consider embedded_degree concept for more accurate degree, see blueprint
        return self._degree

    def quadrature_scheme(self):
        "Return quadrature scheme of finite element"
        return self._quad_scheme

    def mapping(self):
        error("Missing implementation of mapping().")

    def cell(self):
        "Return cell of finite element"
        return self._cell

    def domain(self, component=None): # TODO: Deprecate this
        "Return the domain on which this element is defined."
        domains = self.domains(component)
        n = len(domains)
        if n == 0:
            return None
        elif n == 1:
            return domains[0]
        else:
            error("Cannot return the requested single domain, as this element has multiple domains.")

    def domains(self, component=None):
        "Return the domain on which this element is defined."
        if self._domain is None:
            return ()
        else:
            return (self._domain,)

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

    def symmetry(self): # FIXME: different approach
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1."""
        return EmptyDict

    def _check_component(self, i):
        "Check that component index i is valid"
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j, k) in zip(i, sh))):
            error(("Illegal component index '%r' (value rank %d)" + \
                   "for element (value rank %d).") % (i, len(i), r))

    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative
        component index for a given component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (None, i)

    def extract_component(self, i):
        """Recursively extract component index relative to a (simple) element
        and that element for given value component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (i, self)

    def _check_reference_component(self, i):
        "Check that reference component index i is valid"
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j, k) in zip(i, sh))):
            error(("Illegal component index '%r' (value rank %d)" + \
                   "for element (value rank %d).") % (i, len(i), r))

    def extract_subelement_reference_component(self, i):
        """Extract direct subelement index and subelement relative
        reference component index for a given reference component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_reference_component(i)
        return (None, i)

    def extract_reference_component(self, i):
        """Recursively extract reference component index relative to a (simple) element
        and that element for given reference value component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_reference_component(i)
        return (i, self)

    def num_sub_elements(self):
        "Return number of sub elements"
        return 0

    def sub_elements(self):
        "Return list of sub elements"
        return []

    def __add__(self, other):
        "Add two elements, creating an enriched element"
        ufl_assert(isinstance(other, FiniteElementBase),
                   "Can't add element and %s." % other.__class__)
        warning_blue("WARNING: Creating an EnrichedElement,\n         " +\
                     "if you intended to create a MixedElement use '*' instead of '+'.")
        from ufl.finiteelement import EnrichedElement
        return EnrichedElement(self, other)

    def __mul__(self, other):
        "Multiply two elements, creating a mixed element"
        ufl_assert(isinstance(other, FiniteElementBase),
                   "Can't multiply element and %s." % other.__class__)
        from ufl.finiteelement import MixedElement
        return MixedElement(self, other)

    def __getitem__(self, index):
        "Restrict finite element to a subdomain, subcomponent or topology (cell)."
        # NOTE: RestrictedElement will not be used to represent restriction
        #       to subdomains, as that is represented by the element having
        #       a domain property that is a Region.
        # NOTE: Implementing restriction to subdomains with [] should not be
        #       done, as V[1] is ambiguously similar to both indexing expressions
        #       and obtaining a subdomain, such as myexpr[1] and mydomain[1].
        if isinstance(index, Cell) or index == "facet":
            from ufl.finiteelement import RestrictedElement
            return RestrictedElement(self, index)
        return NotImplemented
