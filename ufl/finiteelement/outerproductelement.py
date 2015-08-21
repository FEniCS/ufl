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
from ufl.cell import OuterProductCell
from ufl.domain import as_domain
from ufl.finiteelement.mixedelement import MixedElement
from ufl.finiteelement.finiteelementbase import FiniteElementBase


class OuterProductElement(FiniteElementBase):
    r"""The outer (tensor) product of 2 element spaces:

    .. math:: V = A \otimes B

    Given bases :math:`{\phi_A, \phi_B}` for :math:`A, B`,
    :math:`{\phi_A * \phi_B}` forms a basis for :math:`V`.
    """
    __slots__ = ("_A", "_B", "_mapping")

    def __init__(self, A, B, domain=None, form_degree=None,
                 quad_scheme=None):
        "Create OuterProductElement from a given pair of elements."
        self._A = A
        self._B = B
        family = "OuterProductElement"

        if domain is None:
            # Define cell as the product of sub-cells
            cell = OuterProductCell(A.cell(), B.cell())
            domain = as_domain(cell)
        else:
            domain = as_domain(domain)
            cell = domain.ufl_cell
            ufl_assert(cell is not None, "Missing cell in given domain.")

        self._repr = "OuterProductElement(*%r, %r)" % (list([self._A, self._B]),
                                                       domain)
        # Define polynomial degree as a tuple of sub-degrees
        degree = (A.degree(), B.degree())

        # match FIAT implementation
        if len(A.value_shape()) == 0 and len(B.value_shape()) == 0:
            value_shape = ()
            reference_value_shape = ()
        elif len(A.value_shape()) == 1 and len(B.value_shape()) == 0:
            value_shape = (A.value_shape()[0],)
            reference_value_shape = (A.reference_value_shape()[0],) # TODO: Is this right?
        elif len(A.value_shape()) == 0 and len(B.value_shape()) == 1:
            value_shape = (B.value_shape()[0],)
            reference_value_shape = (B.reference_value_shape()[0],) # TODO: Is this right?
        else:
            raise Exception("Product of vector-valued elements not supported")

        if A.mapping() == "identity" and B.mapping() == "identity":
            self._mapping = "identity"
        else:
            self._mapping = "undefined"

        FiniteElementBase.__init__(self, family, domain, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def mapping(self):
        return self._mapping

    def reconstruct(self, **kwargs):
        """Construct a new OuterProductElement with some properties
        replaced with new values."""
        domain = kwargs.get("domain", self.domain())
        return OuterProductElement(self._A, self._B, domain=domain)

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data, which must be reconstructed or supplied by other means.
        """
        return "OuterProductElement(%r, %r, %s, %r)" % (
            self._A, self._B, self.domain().reconstruction_signature(),
            self._quad_scheme)

    def __str__(self):
        "Pretty-print."
        return "OuterProductElement(%s)" \
            % str([str(self._A), str(self._B)])

    def shortstr(self):
        "Short pretty-print."
        return "OuterProductElement(%s)" \
            % str([self._A.shortstr(), self._B.shortstr()])

    def signature_data(self, renumbering):
        data = ("OuterProductElement",
                self._A,
                self._B,
                self._quad_scheme,
                ("no domain" if self._domain is None else self._domain.signature_data(renumbering)))
        return data


class OuterProductVectorElement(MixedElement):
    """A special case of a mixed finite element where all
    elements are equal OuterProductElements"""
    __slots__ = ("_sub_element")

    def __init__(self, A, B, domain=None, dim=None,
                 form_degree=None, quad_scheme=None):
        if domain is not None:
            domain = as_domain(domain)

        sub_element = OuterProductElement(A, B, domain=domain)
        dim = dim or sub_element.cell().geometric_dimension()
        sub_elements = [sub_element]*dim

        # Get common family name (checked in FiniteElement.__init__)
        family = sub_element.family()

        # Compute value shape
        shape = (dim,)
        value_shape = shape + sub_element.value_shape()

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = A.degree(), B.degree()

        self._sub_element = sub_element
        # Cache repr string
        self._repr = "OuterProductVectorElement(%r, %r, dim=%d)" % \
            (self._sub_element, self.domain(), len(self._sub_elements))

    @property
    def _A(self):
        return self._sub_element._A

    @property
    def _B(self):
        return self._sub_element._B

    def mapping(self):
        return self._sub_element.mapping()

    def signature_data(self, renumbering):
        data = ("OuterProductVectorElement", self._A, self._B,
                len(self._sub_elements), self._quad_scheme,
                ("no domain" if self._domain is None else
                    self._domain.signature_data(renumbering)))
        return data

    def reconstruct(self, **kwargs):
        """Construct a new OuterProductVectorElement with some properties
        replaced with new values."""
        domain = kwargs.get("domain", self.domain())
        dim = kwargs.get("dim", self.num_sub_elements())
        return OuterProductVectorElement(self._A, self._B,
                                         domain=domain, dim=dim)

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data, which must be reconstructed or supplied by other means.
        """
        return "OuterProductVectorElement(%r, %s, %d, %r)" % (
            self._sub_element, self.domain().reconstruction_signature(),
            len(self._sub_elements), self._quad_scheme)

    def __str__(self):
        "Format as string for pretty printing."
        return "<Outer product vector element: %r x %r>" % \
               (self._sub_element, self.num_sub_elements())

    def shortstr(self):
        "Format as string for pretty printing."
        return "OPVector"
