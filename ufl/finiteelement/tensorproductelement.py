# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from ufl.assertions import ufl_assert
from ufl.cell import as_cell, ProductCell
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.finiteelementbase import FiniteElementBase


class TensorProductElement(FiniteElementBase):
    r"""The tensor product of d element spaces:

    .. math:: V = V_0 \otimes V_1 \otimes ...  \otimes V_d

    Given bases {phi_i} for V_i for i = 1, ...., d,
    { phi_0 * phi_1 * ... * phi_d } forms a basis for V.
    """
    __slots__ = ("_sub_elements",)

    def __init__(self, elements):
        "Create TensorProductElement from a given list of elements."

        warning("The TensorProductElement is work in progress and the design may change at any moment without notice.")

        self._sub_elements = elements
        ufl_assert(len(self._sub_elements) > 0,
                   "Cannot create TensorProductElement from empty list.")
        self._repr = "TensorProductElement(%s)" % ", ".join(repr(e) for e in self._sub_elements)

        family = "TensorProductElement"

        # Define cell as the product of each elements cell
        cell = ProductCell(*[e.cell() for e in self._sub_elements])

        # Define polynomial degree as the maximal of each subelement
        degrees = { e.degree() for e in self._sub_elements } - { None }
        degree = max(degrees) if degrees else None

        # No quadrature scheme defined
        quad_scheme = None

        # For now, check that all subelements have the same value
        # shape, and use this.
        # TODO: Not sure if this makes sense, what kind of product is used to build the basis?
        value_shape = self._sub_elements[0].value_shape()
        reference_value_shape = self._sub_elements[0].reference_value_shape()
        ufl_assert(all(e.value_shape() == value_shape
                       for e in self._sub_elements),
                   "All subelements in must have same value shape")

        FiniteElementBase.__init__(self, family, cell, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def mapping(self):
        if all(e.mapping() == "identity" for e in self._sub_elements):
            return "identity"
        else:
            return "undefined"

    def num_sub_elements(self):
        "Return number of subelements."
        return len(self._sub_elements)

    def sub_elements(self):
        "Return subelements (factors)."
        return self._sub_elements

    def __str__(self):
        "Pretty-print."
        return "TensorProductElement(%s)" \
            % str([str(e) for e in self.sub_elements()])

    def shortstr(self):
        "Short pretty-print."
        return "TensorProductElement(%s)" \
            % str([e.shortstr() for e in self.sub_elements()])
