"""The Integral class."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009
# Modified by Massimiliano Leoni, 2016.

import ufl
from ufl.checks import is_python_scalar, is_scalar_constant_expression
from ufl.core.expr import Expr
from ufl.protocols import id_or_none

# Export list for ufl.classes
__all_classes__ = ["Integral"]


class Integral:
    """An integral over a single domain."""

    __slots__ = (
        "_integral_type",
        "_integrand",
        "_metadata",
        "_subdomain_data",
        "_subdomain_id",
        "_ufl_domain",
    )

    def __init__(self, integrand, integral_type, domain, subdomain_id, metadata, subdomain_data):
        """Initialise."""
        if not isinstance(integrand, Expr):
            raise ValueError("Expecting integrand to be an Expr instance.")
        self._integrand = integrand
        self._integral_type = integral_type
        self._ufl_domain = domain
        self._subdomain_id = subdomain_id
        self._metadata = metadata
        self._subdomain_data = subdomain_data

    def reconstruct(
        self,
        integrand=None,
        integral_type=None,
        domain=None,
        subdomain_id=None,
        metadata=None,
        subdomain_data=None,
    ):
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
            domain = self.ufl_domain()
        if subdomain_id is None:
            subdomain_id = self.subdomain_id()
        if metadata is None:
            metadata = self.metadata()
        if subdomain_data is None:
            subdomain_data = self._subdomain_data
        return Integral(integrand, integral_type, domain, subdomain_id, metadata, subdomain_data)

    def integrand(self):
        """Return the integrand expression, which is an ``Expr`` instance."""
        return self._integrand

    def integral_type(self):
        """Return the domain type of this integral."""
        return self._integral_type

    def ufl_domain(self):
        """Return the integration domain of this integral."""
        return self._ufl_domain

    def subdomain_id(self):
        """Return the subdomain id of this integral."""
        return self._subdomain_id

    def metadata(self):
        """Return the compiler metadata this integral has been annotated with."""
        return self._metadata

    def subdomain_data(self):
        """Return the domain data of this integral."""
        return self._subdomain_data

    def __neg__(self):
        """Negate."""
        return self.reconstruct(-self._integrand)

    def __mul__(self, scalar):
        """Multiply."""
        if not is_python_scalar(scalar):
            raise ValueError("Cannot multiply an integral with non-constant values.")
        return self.reconstruct(scalar * self._integrand)

    def __rmul__(self, scalar):
        """Multiply."""
        if not is_scalar_constant_expression(scalar):
            raise ValueError(
                "An integral can only be multiplied by a globally constant scalar expression."
            )
        return self.reconstruct(scalar * self._integrand)

    def __str__(self):
        """Format as a string."""
        fmt = "{ %s } * %s(%s[%s], %s)"
        mname = ufl.measure.integral_type_to_measure_name[self._integral_type]
        s = fmt % (self._integrand, mname, self._ufl_domain, self._subdomain_id, self._metadata)
        return s

    def __repr__(self):
        """Representation."""
        return (
            f"Integral({self._integrand!r}, {self._integral_type!r}, {self._ufl_domain!r}, "
            f"{self._subdomain_id!r}, {self._metadata!r}, {self._subdomain_data!r})"
        )

    def __eq__(self, other):
        """Check equality."""
        return (
            isinstance(other, Integral)
            and self._integral_type == other._integral_type
            and self._ufl_domain == other._ufl_domain
            and self._subdomain_id == other._subdomain_id
            and self._integrand == other._integrand
            and self._metadata == other._metadata
            and id_or_none(self._subdomain_data) == id_or_none(other._subdomain_data)
        )

    def __hash__(self):
        """Hash."""
        # Assuming few collisions by ignoring hash(self._metadata) (a
        # dict is not hashable but we assume it is immutable in
        # practice)
        hashdata = (
            hash(self._integrand),
            self._integral_type,
            hash(self._ufl_domain),
            self._subdomain_id,
            id_or_none(self._subdomain_data),
        )
        return hash(hashdata)
