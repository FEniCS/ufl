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


class TraceElement(FiniteElementBase):
    """A finite element space: the trace of a given hdiv element.
    This is effectively a scalar-valued restriction which is
    non-zero only on cell facets."""
    def __init__(self, element):
        self._element = element
        self._repr = "TraceElement(%s)" % str(element._repr)

        family = "TraceElement"
        domain = element.domain()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = ()
        reference_value_shape = ()
        FiniteElementBase.__init__(self, family, domain, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def reconstruct(self, **kwargs):
        """Construct a new TraceElement object with some properties
        replaced with new values."""
        domain = kwargs.get("domain", self.domain())
        ele = self._element.reconstruct(domain=domain)
        return TraceElement(ele)

    def reconstruction_signature(self):
        return "TraceElement(%s)" % self._element.reconstruction_signature()

    def signature_data(self, renumbering):
        data = ("TraceElement", self._element.signature_data(renumbering),
                ("no domain" if self._domain is None else self._domain
                    .signature_data(renumbering)))
        return data

    def __str__(self):
        return "TraceElement(%s)" % str(self._element)

    def shortstr(self):
        return "TraceElement(%s)" % str(self._element.shortstr())

    def __repr__(self):
        return self._repr
