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
# Modified by Andrew T. T. McRae 2014
# Modified by Lawrence Mitchell 2014
# Modified by Marie E. Rognes 2010, 2012

from ufl.assertions import ufl_assert
from ufl.cell import TensorProductCell, as_cell
from ufl.finiteelement.finiteelementbase import FiniteElementBase


class TensorProductElement(FiniteElementBase):
    r"""The outer (tensor) product of 2 element spaces:

    .. math:: V = A \otimes B

    Given bases :math:`{\phi_A, \phi_B}` for :math:`A, B`,
    :math:`{\phi_A \otimes \phi_B}` forms a basis for :math:`V`.
    """
    __slots__ = ("_A", "_B", "_mapping")

    def __init__(self, A, B, cell=None):
        "Create TensorProductElement from a given pair of elements."
        self._A = A
        self._B = B
        family = "TensorProductElement"

        if cell is None:
            # Define cell as the product of sub-cells
            cell = TensorProductCell(A.cell(), B.cell())
        else:
            cell = as_cell(cell)

        self._repr = "TensorProductElement(%r, %r, %r)" % (self._A, self._B, cell)

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
        return "TensorProductElement(%s)" \
            % str([str(self._A), str(self._B)])

    def shortstr(self):
        "Short pretty-print."
        return "TensorProductElement(%s)" \
            % str([self._A.shortstr(), self._B.shortstr()])
