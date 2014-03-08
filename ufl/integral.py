"""The Integral class."""

# Copyright (C) 2008-2014 Martin Sandve Alnes
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
# Modified by Anders Logg, 2008-2009

import ufl
from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.checks import (is_true_ufl_scalar, is_python_scalar, is_globally_constant,
                        is_scalar_constant_expression)
from ufl.measure import Measure
from ufl.protocols import id_or_none

class Integral(object):
    "An integral over a single domain."
    __slots__ = ("_integrand",
                 "_domain_type",
                 "_domain",
                 "_domain_id",
                 "_metadata",
                 "_domain_data",
                 )
    def __init__(self, integrand, domain_type, domain, domain_id, metadata, domain_data):
        ufl_assert(isinstance(integrand, Expr),
                   "Expecting integrand to be an Expr instance.")
        self._integrand = integrand
        self._domain_type = domain_type
        self._domain = domain
        self._domain_id = domain_id
        self._metadata = metadata
        self._domain_data = domain_data

    def reconstruct(self, integrand=None,
                    domain_type=None, domain=None, domain_id=None,
                    metadata=None, domain_data=None):
        """Construct a new Integral object with some properties replaced with new values.

        Example:
            <a = Integral instance>
            b = a.reconstruct(expand_compounds(a.integrand()))
            c = a.reconstruct(metadata={'quadrature_degree':2})
        """
        if integrand is None:
            integrand = self.integrand()
        if domain_type is None:
            domain_type = self.domain_type()
        if domain is None:
            domain = self.domain()
        if domain_id is None:
            domain_id = self.domain_id()
        if metadata is None:
            metadata = self.metadata()
        if domain_data is None:
            domain_data = self._domain_data
        return Integral(integrand, domain_type, domain, domain_id, metadata, domain_data)

    def integrand(self):
        "Return the integrand expression, which is an Expr instance."
        return self._integrand

    def domain_type(self):
        "Return the domain type of this integral."
        return self._domain_type

    def domain(self):
        "Return the integration domain of this integral."
        return self._domain

    def domain_id(self):
        "Return the domain id of this integral."
        return self._domain_id

    def metadata(self):
        "Return the compiler metadata this integral has been annotated with."
        return self._metadata

    def domain_data(self):
        "Return the domain data of this integral."
        return self._domain_data

    def __neg__(self):
        return self.reconstruct(-self._integrand)

    def __mul__(self, scalar):
        ufl_assert(is_python_scalar(scalar),
                   "Cannot multiply an integral with non-constant values.")
        return self.reconstruct(scalar*self._integrand)

    def __rmul__(self, scalar):
        ufl_assert(is_scalar_constant_expression(scalar),
                   "An integral can only be multiplied by a "
                   "globally constant scalar expression.")
        return self.reconstruct(scalar*self._integrand)

    def __str__(self):
        fmt = "{ %s } * %s(%s[%s], %s)"
        mname = ufl.measure.domain_type_to_measure_name[self._domain_type]
        s = fmt % (self._integrand, mname, self._domain, self._domain_id, self._metadata)
        return s

    def __repr__(self):
        return "Integral(%r, %r, %r, %r, %r, %r)" % (
            self._integrand, self._domain_type, self._domain, self._domain_id, self._metadata, self._domain_data)

    def __eq__(self, other):
        return (isinstance(other, Integral)
            and self._domain_type == other._domain_type
            and self._domain == other._domain
            and self._domain_id == other._domain_id
            and self._integrand == other._integrand
            and self._metadata == other._metadata
            and id_or_none(self._domain_data) == id_or_none(other._domain_data))

    def __hash__(self):
        # Assuming few collisions by ignoring hash(self._metadata)
        # (a dict is not hashable but we assume it is immutable in practice)
        hashdata = (hash(self._integrand), self._domain_type,
                    hash(self._domain), self._domain_id,
                    id_or_none(self._domain_data))
        return hash(hashdata)

