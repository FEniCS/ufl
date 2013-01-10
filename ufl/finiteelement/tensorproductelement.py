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
from ufl.geometry import as_cell, ProductCell
from ufl.domains import as_domain
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.finiteelementbase import FiniteElementBase


class TensorProductElement(FiniteElementBase):
    r"""The tensor product of d element spaces:

    .. math:: V = V_0 \otimes V_1 \otimes ...  \otimes V_d

    Given bases {phi_i} for V_i for i = 1, ...., d,
    { phi_0 * phi_1 * ... * phi_d } forms a basis for V.
    """
    __slots__ = ("_sub_elements",)

    def __init__(self, *elements):
        "Create TensorProductElement from a given list of elements."

        self._sub_elements = list(elements)
        ufl_assert(len(self._sub_elements) > 0,
                   "Cannot create TensorProductElement from empty list.")
        self._repr = "TensorProductElement(*%r)" % self._sub_elements
        family = "TensorProductElement"

        # Define cell as the product of each subcell
        cell = ProductCell(*[e.cell() for e in self._sub_elements])
        domain = as_domain(cell) # FIXME: figure out what this is supposed to mean :)

        # Define polynomial degree as the maximal of each subelement
        degree = max(e.degree() for e in self._sub_elements)

        # No quadrature scheme defined
        quad_scheme = None

        # For now, check that all subelements have the same value
        # shape, and use this.
        # TODO: Not sure if this makes sense, what kind of product is used to build the basis?
        value_shape = self._sub_elements[0].value_shape()
        ufl_assert(all(e.value_shape() == value_shape
                       for e in self._sub_elements),
                   "All subelements in must have same value shape")

        super(TensorProductElement, self).__init__(family, domain, degree,
                                                   quad_scheme, value_shape)

    def num_sub_elements(self):
        "Return number of subelements."
        return len(self._sub_elements)

    def sub_elements(self):
        "Return subelements (factors)."
        # TODO: I don't think the concept of sub elements is quite well defined across
        # all the current element types. Need to investigate how sub_elements is used in
        # existing code, and eventually redesign here, hopefully just adding another function.
        # Summary of different behaviours:
        # - In MixedElement and subclasses each subelement corresponds to different value components.
        # - In EnrichedElement sub_elements returns [] even though it contains multiple "children".
        # - In RestrictedElement it returns the sub elements of its single children.
        # - Here in TensorProductElement it returns sub elements that correspond to factors, not components.
        return self._sub_elements

    def num_tensorproduct_sub_elements(self): # FIXME: Use this where intended, for disambiguation w.r.t. different sub_elements meanings.
        "Return number of tensorproduct sub elements."
        return len(self._sub_elements)

    def tensorproduct_sub_elements(self): # FIXME: Use this where intended, for disambiguation w.r.t. different sub_elements meanings.
        "Return list of tensorproduct sub elements."
        return self._sub_elements

    def __str__(self):
        "Pretty-print."
        return "TensorProductElement(%s)" \
            % str([str(e) for e in self.sub_elements()])

    def shortstr(self):
        "Short pretty-print."
        return "TensorProductElement(%s)" \
            % str([e.shortstr() for e in self.sub_elements()])

    def __repr__(self):
        return self._repr
