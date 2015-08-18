# Copyright (C) 2008-2014 Andrew T. T. McRae
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

from ufl.finiteelement.outerproductelement import OuterProductElement
from ufl.finiteelement.finiteelementbase import FiniteElementBase


class HDivElement(OuterProductElement):
    """A div-conforming version of an outer product element, assuming
    this makes mathematical sense."""
    __slots__ = ("_element")

    def __init__(self, element):
        self._element = element
        self._repr = "HDivElement(%r)" % element
        self._mapping = "contravariant Piola"

        family = "OuterProductElement"
        domain = element.domain()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = (element.cell().geometric_dimension(),)
        reference_value_shape = (element.cell().topological_dimension(),) # TODO: Is this right?
        # Skipping OuterProductElement constructor! Bad code smell, refactor to avoid this non-inheritance somehow.
        FiniteElementBase.__init__(self, family, domain, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def reconstruct(self, **kwargs):
        """Construct a new HDivElement object with some properties
        replaced with new values."""
        domain = kwargs.get("domain", self.domain())
        ele = self._element.reconstruct(domain=domain)
        return HDivElement(ele)

    def reconstruction_signature(self):
        return "HDivElement(%s)" % self._element.reconstruction_signature()

    def signature_data(self, renumbering):
        data = ("HDivElement", self._element.signature_data(renumbering),
                ("no domain" if self._domain is None else self._domain
                    .signature_data(renumbering)))
        return data

    def __str__(self):
        return "HDivElement(%s)" % str(self._element)

    def shortstr(self):
        return "HDivElement(%s)" % str(self._element.shortstr())

    def __repr__(self):
        return self._repr


class HCurlElement(OuterProductElement):
    """A curl-conforming version of an outer product element, assuming
    this makes mathematical sense."""
    __slots__ = ("_element")

    def __init__(self, element):
        self._element = element
        self._repr = "HCurlElement(%r)" % element
        self._mapping = "covariant Piola"

        family = "OuterProductElement"
        domain = element.domain()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = (element.cell().geometric_dimension(),)
        reference_value_shape = (element.cell().topological_dimension(),) # TODO: Is this right?
        # Skipping OuterProductElement constructor! Bad code smell, refactor to avoid this non-inheritance somehow.
        FiniteElementBase.__init__(self, family, domain, degree,
                                    quad_scheme, value_shape, reference_value_shape)

    def reconstruct(self, **kwargs):
        """Construct a new HCurlElement object with some properties
        replaced with new values."""
        domain = kwargs.get("domain", self.domain())
        ele = self._element.reconstruct(domain=domain)
        return HCurlElement(ele)

    def reconstruction_signature(self):
        return "HCurlElement(%s)" % self._element.reconstruction_signature()

    def signature_data(self, renumbering):
        data = ("HCurlElement", self._element.signature_data(renumbering),
                ("no domain" if self._domain is None else self._domain
                    .signature_data(renumbering)))
        return data

    def __str__(self):
        return "HCurlElement(%s)" % str(self._element)

    def shortstr(self):
        return "HCurlElement(%s)" % str(self._element.shortstr())

    def __repr__(self):
        return self._repr
