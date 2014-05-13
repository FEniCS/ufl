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


class TraceSpace(FiniteElementBase):
    """A finite element space: the trace of a given hdiv element.
    This is effectively a scalar-valued restriction which is
    non-zero only on cell facets."""
    def __init__(self, element):
        self._element = element
        self._repr = "Trace(%s)" % str(element._repr)

        family = "Trace"
        domain = element.domain()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = ()
        super(TraceSpace, self).__init__(family, domain, degree,
                                         quad_scheme, value_shape)

    def reconstruct(self, **kwargs):
        """Construct a new TraceSpace object with some properties
        replaced with new values."""
        domain = kwargs.get("domain", self.domain())
        ele = self._element.reconstruct(domain=domain)
        return TraceSpace(ele)

    def reconstruction_signature(self):
        return "Trace(%s)" % self._element.reconstruction_signature()

    def signature_data(self, domain_numbering):
        data = ("Trace", self._element.signature_data(domain_numbering=domain_numbering),
                ("no domain" if self._domain is None else self._domain
                    .signature_data(domain_numbering=domain_numbering)))
        return data

    def __str__(self):
        return "Trace(%s)" % str(self._element)

    def shortstr(self):
        return "Trace(%s)" % str(self._element.shortstr())

    def __repr__(self):
        return self._repr
