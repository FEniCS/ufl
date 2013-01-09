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
# Last changed: 2012-08-16

from itertools import izip
from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.common import product, index_to_component, component_to_index, istr, EmptyDict
from ufl.geometry import as_cell, cellname2facetname, ProductCell
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.elementlist import ufl_elements, aliases

class FiniteElementBase(object):
    "Base class for all finite elements"
    __slots__ = ("_family", "_cell", "_degree", "_quad_scheme", "_value_shape",
                 "_repr", "_domain")

    def __init__(self, family, cell, degree, quad_scheme, value_shape):
        "Initialize basic finite element data"
        ufl_assert(isinstance(family, str), "Invalid family type.")
        cell = as_cell(cell)
        ufl_assert(isinstance(degree, int) or degree is None, "Invalid degree type.")
        ufl_assert(isinstance(value_shape, tuple), "Invalid value_shape type.")
        self._family = family
        self._cell = cell
        self._degree = degree
        self._value_shape = value_shape
        self._domain = None
        self._quad_scheme = quad_scheme

    def unique_basic_elements(self):
        return [] # FIXME: Return list of unique basic elements for all subclasses

    def basic_element_instances(self):
        return [] # FIXME: Return list of non-unique basic elements for all subclasses, or list of indices into unique_basic_elements?

    def family(self): # FIXME: Undefined for base?
        "Return finite element family"
        return self._family

    def cell(self):
        "Return cell of finite element"
        return self._cell

    def is_cellwise_constant(self): # FIXME: Per component?
        "Return whether the basis functions of this element is spatially constant over each cell."
        return self.family() == "Real" or self.degree() == 0

    def degree(self): # FIXME: Per component?
        "Return polynomial degree of finite element"
        return self._degree

    def quadrature_scheme(self):
        "Return quadrature scheme of finite element"
        return self._quad_scheme

    def value_shape(self):
        "Return the shape of the value space"
        return self._value_shape

    def symmetry(self): # FIXME: different approach
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1."""
        return EmptyDict

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

    def domain_restriction(self):
        "Return the domain onto which the element is restricted."
        return self._domain

    def num_sub_elements(self):
        "Return number of sub elements"
        return 0

    def sub_elements(self): # FIXME: Replace with alternative variants
        "Return list of sub elements"
        return []

    def _check_component(self, i):
        "Check that component index i is valid"
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j,k) in izip(i, sh))):
            error(("Illegal component index '%r' (value rank %d)" + \
                   "for element (value rank %d).") % (i, len(i), r))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __add__(self, other):
        "Add two elements, creating an enriched element"
        ufl_assert(isinstance(other, FiniteElementBase), "Can't add element and %s." % other.__class__)
        warning_blue("WARNING: Creating an EnrichedElement,\n         " +\
                     "if you intended to create a MixedElement use '*' instead of '+'.")
        from ufl.finiteelement import EnrichedElement
        return EnrichedElement(self, other)

    def __repr__(self):
        "Format as string for evaluation as Python object."
        return self._repr

    def __mul__(self, other):
        "Multiply two elements, creating a mixed element"
        ufl_assert(isinstance(other, FiniteElementBase), "Can't multiply element and %s." % other.__class__)
        from ufl.finiteelement import MixedElement
        return MixedElement(self, other)

    def __getitem__(self, index):
        "Restrict finite element to a subdomain, subcomponent or topology (cell)."
        from ufl.integral import Measure
        from ufl.geometry import Cell
        if isinstance(index, (Measure, Cell)) or\
                index == "facet" or\
                isinstance(as_cell(index), Cell): # TODO: Can we just drop the as_cell call?
            from ufl.finiteelement import RestrictedElement
            return RestrictedElement(self, index)
        #if isinstance(index, int):
        #    return SubElement(self, index)
        return NotImplemented
