"This module defines the UFL finite element classes."

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
#
# First added:  2008-03-03
# Last changed: 2013-01-10

from itertools import izip
from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.common import product, index_to_component, component_to_index, istr, EmptyDict
from ufl.geometry import Cell, as_cell, cellname2facetname, ProductCell
from ufl.domains import as_domain, DomainDescription
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.elementlist import ufl_elements, aliases

class FiniteElementBase(object):
    "Base class for all finite elements"
    __slots__ = ("_cell", "_domain", "_family", "_degree",
                 "_quad_scheme", "_value_shape", "_repr",)

    def __init__(self, family, domain, degree, quad_scheme, value_shape):
        "Initialize basic finite element data"
        ufl_assert(isinstance(family, str), "Invalid family type.")
        ufl_assert(isinstance(degree, int) or degree is None, "Invalid degree type.")
        ufl_assert(isinstance(value_shape, tuple), "Invalid value_shape type.")
        if domain is None:
            self._domain = None
            self._cell = None
        else:
            self._domain = as_domain(domain)
            self._cell = self._domain.cell()
            ufl_assert(isinstance(self._domain, DomainDescription), "Invalid domain type.")
            ufl_assert(isinstance(self._cell, Cell), "Invalid cell type.")
        self._family = family
        self._degree = degree
        self._value_shape = value_shape
        self._quad_scheme = quad_scheme

    def __repr__(self):
        "Format as string for evaluation as Python object."
        return self._repr

    def __hash__(self):
        "Compute hash code for insertion in hashmaps."
        return hash(repr(self))

    def __eq__(self, other):
        "Compute element equality for insertion in hashmaps."
        return type(self) == type(other) and repr(self) == repr(other)

    def __lt__(self, other):
        "Compare elements by repr, to give a natural stable sorting."
        return repr(self) < repr(other)

    def domain(self, component=None):
        "Return the domain on which this element is defined."
        return self._domain

    def regions(self):
        "Return the regions referenced by this element and its subelements."
        return [self._domain] # FIXME

    def cell_restriction(self):
        "Return the cell type onto which the element is restricted."
        return None # Overloaded by RestrictedElement

    def cell(self):
        "Return cell of finite element"
        return self._cell

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

    def is_cellwise_constant(self, component=None):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        return self.family() == "Real" or self.degree() == 0

    def value_shape(self):
        "Return the shape of the value space"
        return self._value_shape

    def symmetry(self): # FIXME: different approach
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1."""
        return EmptyDict

    def unique_basic_elements(self):
        # FIXME: Return list of unique basic elements for all subclasses
        return []

    def basic_element_instances(self):
        # FIXME: Return list of non-unique basic elements for all subclasses,
        #        or list of indices into unique_basic_elements?
        return []

    def _check_component(self, i):
        "Check that component index i is valid"
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j,k) in izip(i, sh))):
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

    def num_sub_elements(self):
        "Return number of sub elements"
        return 0

    def sub_elements(self): # FIXME: Replace with alternative variants
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
