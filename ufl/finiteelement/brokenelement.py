# Copyright (C) 2014 Andrew T. T. McRae
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

from ufl.finiteelement.finiteelementbase import FiniteElementBase


class BrokenElement(FiniteElementBase):
    """The discontinuous version of an existing Finite Element space"""
    def __init__(self, element):
        self._element = element
        self._repr = "BrokenElement(%s)" % str(element._repr)

        family = "BrokenElement"
        domain = element.domain()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = element.value_shape()
        super(BrokenElement, self).__init__(family, domain, degree,
                                         quad_scheme, value_shape)

    def reconstruct(self, **kwargs):
        """Construct a new BrokenElement object with some properties
        replaced with new values."""
        domain = kwargs.get("domain", self.domain())
        ele = self._element.reconstruct(domain=domain)
        return BrokenElement(ele)

    def reconstruction_signature(self):
        return "BrokenElement(%s)" % self._element.reconstruction_signature()

    def signature_data(self, renumbering):
        data = ("BrokenElement", self._element.signature_data(renumbering),
                ("no domain" if self._domain is None else self._domain
                    .signature_data(renumbering)))
        return data

    def __str__(self):
        return "BrokenElement(%s)" % str(self._element)

    def shortstr(self):
        return "BrokenElement(%s)" % str(self._element.shortstr())

    def __repr__(self):
        return self._repr
