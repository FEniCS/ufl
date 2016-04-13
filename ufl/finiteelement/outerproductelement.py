# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

# Copyright (C) 2008-2015 Martin Sandve Alnæs and Andrew T. T. McRae
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
# Based on tensorproductelement.py
# Modified by Andrew T. T. McRae 2014
# Modified by Lawrence Mitchell 2014
# Modified by Massimiliano Leoni, 2016

from ufl.assertions import ufl_assert
from ufl.cell import OuterProductCell, as_cell
from ufl.finiteelement.mixedelement import MixedElement
from ufl.finiteelement.finiteelementbase import FiniteElementBase


class OuterProductElement(FiniteElementBase):
    r"""The outer (tensor) product of 2 element spaces:

    .. math:: V = A \otimes B

    Given bases :math:`{\phi_A, \phi_B}` for :math:`A, B`,
    :math:`{\phi_A \otimes \phi_B}` forms a basis for :math:`V`.
    """
    __slots__ = ("_A", "_B", "_mapping")

    def __init__(self, A, B, cell=None):
        "Create OuterProductElement from a given pair of elements."
        self._A = A
        self._B = B
        family = "OuterProductElement"

        if cell is None:
            # Define cell as the product of sub-cells
            cell = OuterProductCell(A.cell(), B.cell())
        else:
            cell = as_cell(cell)

        self._repr = "OuterProductElement(%r, %r, %r)" % (self._A, self._B, cell)

        # Define polynomial degree as a tuple of sub-degrees
        degree = (A.degree(), B.degree())

        # match FIAT implementation
        value_shape = A.value_shape() + B.value_shape()
        reference_value_shape = A.reference_value_shape() + B.reference_value_shape()
        ufl_assert(len(value_shape) <= 1, "Product of vector-valued elements not supported")
        ufl_assert(len(reference_value_shape) <= 1, "Product of vector-valued elements not supported")

        if A.mapping() == "identity" and B.mapping() == "identity":
            self._mapping = "identity"
        else:
            self._mapping = "undefined"

        FiniteElementBase.__init__(self, family, cell, degree,
                                   None, value_shape, reference_value_shape)

    def mapping(self):
        return self._mapping

    def __str__(self):
        "Pretty-print."
        return "OuterProductElement(%s)" \
            % str([str(self._A), str(self._B)])

    def shortstr(self):
        "Short pretty-print."
        return "OuterProductElement(%s)" \
            % str([self._A.shortstr(), self._B.shortstr()])


class OuterProductVectorElement(MixedElement):
    """A special case of a mixed finite element where all
    elements are equal ``OuterProductElement``s."""
    __slots__ = ("_sub_element")

    def __init__(self, A, B, cell=None, dim=None):
        if cell is not None:
            cell = as_cell(cell)

        sub_element = OuterProductElement(A, B, cell=cell)
        dim = dim or sub_element.cell().geometric_dimension()
        sub_elements = [sub_element]*dim

        # Get common family name (checked in FiniteElement.__init__)
        family = sub_element.family()

        # Compute value shape
        value_shape = (dim,) + sub_element.value_shape()

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = (A.degree(), B.degree())

        self._sub_element = sub_element

        # Cache repr string
        self._repr = "OuterProductVectorElement(%r, %r, dim=%d)" % \
            (self._sub_element, self.cell(), len(self._sub_elements))

    @property
    def _A(self):
        return self._sub_element._A

    @property
    def _B(self):
        return self._sub_element._B

    def mapping(self):
        return self._sub_element.mapping()

    def __str__(self):
        "Format as string for pretty printing."
        return "<Outer product vector element: %r x %r>" % \
               (self._sub_element, self.num_sub_elements())

    def shortstr(self):
        "Format as string for pretty printing."
        return "OPVector"
