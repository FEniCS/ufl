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

from ufl.assertions import ufl_assert
from ufl.cell import TensorProductCell, as_cell
from ufl.finiteelement.mixedelement import MixedElement, _tensor_sub_elements
from ufl.finiteelement.finiteelementbase import FiniteElementBase


class TensorProductElement(FiniteElementBase):
    r"""The outer (tensor) product of 2 element spaces:

    .. math:: V = A \otimes B

    Given bases :math:`{\phi_A, \phi_B}` for :math:`A, B`,
    :math:`{\phi_A * \phi_B}` forms a basis for :math:`V`.
    """
    __slots__ = ("_A", "_B", "_mapping")

    def __init__(self, A, B, cell=None):
        "Create OuterProductElement from a given pair of elements."
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


class TensorProductVectorElement(MixedElement):
    """A special case of a mixed finite element where all
    elements are equal TensorProductElements"""
    __slots__ = ("_sub_element")

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], TensorProductElement):
            self._from_sub_element(*args, **kwargs)
        else:
            self._from_product_parts(*args, **kwargs)

    def _from_product_parts(self, A, B, cell=None, dim=None):
        sub_element = TensorProductElement(A, B, cell=cell)
        self._from_sub_element(sub_element, dim=dim)

    def _from_sub_element(self, sub_element, dim=None):
        assert isinstance(sub_element, TensorProductElement)

        dim = dim or sub_element.cell().geometric_dimension()
        sub_elements = [sub_element]*dim

        # Get common family name (checked in FiniteElement.__init__)
        family = sub_element.family()

        # Compute value shape
        value_shape = (dim,) + sub_element.value_shape()

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = sub_element.degree()

        self._sub_element = sub_element

        # Cache repr string
        self._repr = "TensorProductVectorElement(%r, dim=%d)" % \
            (self._sub_element, len(self._sub_elements))

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


class TensorProductTensorElement(MixedElement):
    """A special case of a mixed finite element where all
    elements are equal TensorProductElements"""
    __slots__ = ("_sub_element", "_shape", "_symmetry",
                 "_sub_element_mapping", "_flattened_sub_element_mapping",
                 "_mapping")

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], TensorProductElement):
            self._from_sub_element(*args, **kwargs)
        else:
            self._from_product_parts(*args, **kwargs)

    def _from_product_parts(self, A, B, cell=None,
                            shape=None, symmetry=None, quad_scheme=None):
        sub_element = TensorProductElement(A, B, cell=cell,
                                          quad_scheme=quad_scheme)
        self._from_sub_element(sub_element, shape=shape, symmetry=symmetry)

    def _from_sub_element(self, sub_element, shape=None, symmetry=None):
        assert isinstance(sub_element, TensorProductElement)

        shape, symmetry, sub_elements, sub_element_mapping, flattened_sub_element_mapping, \
          reference_value_shape, mapping = _tensor_sub_elements(sub_element, shape, symmetry)

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=shape,
                              reference_value_shape=reference_value_shape)
        self._family = sub_element.family()
        self._degree = sub_element.degree()
        self._sub_element = sub_element
        self._shape = shape
        self._symmetry = symmetry
        self._sub_element_mapping = sub_element_mapping
        self._flattened_sub_element_mapping = flattened_sub_element_mapping
        self._mapping = mapping

        # Cache repr string
        self._repr = "TensorProductTensorElement(%r, shape=%r, symmetry=%r)" % \
            (self._sub_element, self._shape, self._symmetry)

    @property
    def _A(self):
        return self._sub_element._A

    @property
    def _B(self):
        return self._sub_element._B

    def signature_data(self, renumbering):
        data = ("TensorProductTensorElement", self._A, self._B,
                self._shape, self._symmetry, self._quad_scheme,
                ("no cell" if self._cell is None else
                    self._cell.signature_data(renumbering)))
        return data

    def reconstruct(self, **kwargs):
        """Construct a new TensorProductTensorElement with some properties
        replaced with new values."""
        cell = kwargs.get("cell", self.cell())
        shape = kwargs.get("shape", self._shape)
        symmetry = kwargs.get("symmetry", self._symmetry)
        return TensorProductTensorElement(self._A, self._B, cell=cell,
                                         shape=shape, symmetry=symmetry)

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data, which must be reconstructed or supplied by other means.
        """
        return "TensorProductTensorElement(%r, %r, %r, %r)" % (
            self._sub_element, self._shape,
            self._symmetry, self._quad_scheme)

    def __str__(self):
        "Format as string for pretty printing."
        return "<Outer product tensor element: %r x %r>" % \
               (self._sub_element, self._shape)

    def shortstr(self):
        "Format as string for pretty printing."
        return "OPTensor"
