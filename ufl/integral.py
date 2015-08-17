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
from ufl.core.expr import Expr
from ufl.checks import (is_true_ufl_scalar, is_python_scalar, is_globally_constant,
                        is_scalar_constant_expression)
from ufl.measure import Measure
from ufl.protocols import id_or_none

class Integral(object):
    "An integral over a single domain."
    __slots__ = ("_integrand",
                 "_integral_type",
                 "_domain",
                 "_subdomain_id",
                 "_metadata",
                 "_subdomain_data",
                 )
    def __init__(self, integrand, integral_type, domain, subdomain_id, metadata, subdomain_data):
        ufl_assert(isinstance(integrand, Expr),
                   "Expecting integrand to be an Expr instance.")
        self._integrand = integrand
        self._integral_type = integral_type
        self._domain = domain
        self._subdomain_id = subdomain_id
        self._metadata = metadata
        self._subdomain_data = subdomain_data

    def reconstruct(self, integrand=None,
                    integral_type=None, domain=None, subdomain_id=None,
                    metadata=None, subdomain_data=None):
        """Construct a new Integral object with some properties replaced with new values.

        Example:
            <a = Integral instance>
            b = a.reconstruct(expand_compounds(a.integrand()))
            c = a.reconstruct(metadata={'quadrature_degree':2})
        """
        if integrand is None:
            integrand = self.integrand()
        if integral_type is None:
            integral_type = self.integral_type()
        if domain is None:
            domain = self.domain()
        if subdomain_id is None:
            subdomain_id = self.subdomain_id()
        if metadata is None:
            metadata = self.metadata()
        if subdomain_data is None:
            subdomain_data = self._subdomain_data
        return Integral(integrand, integral_type, domain, subdomain_id, metadata, subdomain_data)

    def integrand(self):
        "Return the integrand expression, which is an Expr instance."
        return self._integrand

    def integral_type(self):
        "Return the domain type of this integral."
        return self._integral_type

    def domain(self):
        "Return the integration domain of this integral."
        return self._domain

    def subdomain_id(self):
        "Return the subdomain id of this integral."
        return self._subdomain_id

    def metadata(self):
        "Return the compiler metadata this integral has been annotated with."
        return self._metadata

    def subdomain_data(self):
        "Return the domain data of this integral."
        return self._subdomain_data

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
        mname = ufl.measure.integral_type_to_measure_name[self._integral_type]
        s = fmt % (self._integrand, mname, self._domain, self._subdomain_id, self._metadata)
        return s

    def __repr__(self):
        return "Integral(%r, %r, %r, %r, %r, %r)" % (
            self._integrand, self._integral_type, self._domain, self._subdomain_id, self._metadata, self._subdomain_data)

    def __eq__(self, other):
        return (isinstance(other, Integral)
            and self._integral_type == other._integral_type
            and self._domain == other._domain
            and self._subdomain_id == other._subdomain_id
            and self._integrand == other._integrand
            and self._metadata == other._metadata
            and id_or_none(self._subdomain_data) == id_or_none(other._subdomain_data))

    def __hash__(self):
        # Assuming few collisions by ignoring hash(self._metadata)
        # (a dict is not hashable but we assume it is immutable in practice)
        hashdata = (hash(self._integrand), self._integral_type,
                    hash(self._domain), self._subdomain_id,
                    id_or_none(self._subdomain_data))
        return hash(hashdata)
