"""The Integral class."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009
# Modified by Massimiliano Leoni, 2016.

from __future__ import annotations

from typing import Any

import ufl
from ufl.checks import is_python_scalar, is_scalar_constant_expression
from ufl.core.expr import Expr
from ufl.domain import sort_domains
from ufl.protocols import id_or_none

# Export list for ufl.classes
__all_classes__ = ["Integral"]


class Integral:
    """An integral over a single domain."""

    __slots__ = (
        "_extra_domain_integral_type_map",
        "_integral_type",
        "_integrand",
        "_metadata",
        "_subdomain_data",
        "_subdomain_id",
        "_ufl_domain",
    )

    def __init__(
        self,
        integrand: Expr,
        integral_type: str,
        domain,
        subdomain_id: int | tuple[int, ...],
        metadata: dict,
        subdomain_data: Any,
        extra_domain_integral_type_map: dict | None = None,
    ):
        """Initialise.

        Args:
            integrand: Integrand of the integral
            integral_type: Type of integral, see: `ufl.measure._integral_types` for available types.
            domain: Integration domain as an `ufl.Mesh`.
            subdomain_id: Integer or tuple of integer restricting the integral to a subset of
                entities, marked in some way through `subdomain_data`.
            metadata: Metadata that is usually passed to the form compiler
            subdomain_data: Data associated with the subdomain, can be anything
                (depends on the compiler and user-facing API)
            extra_domain_integral_type_map: Mapping from other `ufl.Mesh` objects present in
                `integrand` to specific integral types on this domain.
        """
        if not isinstance(integrand, Expr):
            raise ValueError("Expecting integrand to be an Expr instance.")
        self._integrand = integrand
        self._integral_type = integral_type
        self._ufl_domain = domain
        self._subdomain_id = subdomain_id
        self._metadata = metadata
        self._subdomain_data = subdomain_data
        if extra_domain_integral_type_map is None:
            self._extra_domain_integral_type_map = {}
        else:
            self._extra_domain_integral_type_map = {
                d: extra_domain_integral_type_map[d]
                for d in sort_domains(tuple(extra_domain_integral_type_map.keys()))
            }

    def reconstruct(
        self,
        integrand=None,
        integral_type=None,
        domain=None,
        subdomain_id=None,
        metadata=None,
        subdomain_data=None,
        extra_domain_integral_type_map=None,
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
        if extra_domain_integral_type_map is None:
            extra_domain_integral_type_map = self._extra_domain_integral_type_map
        return Integral(
            integrand,
            integral_type,
            domain,
            subdomain_id,
            metadata,
            subdomain_data,
            extra_domain_integral_type_map=extra_domain_integral_type_map,
        )

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

    def extra_domain_integral_type_map(self):
        """Return the extra domain-integral_type map."""
        return self._extra_domain_integral_type_map

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
        mname = ufl.measure.integral_type_to_measure_name[self._integral_type]
        temp = {
            d: ufl.measure.integral_type_to_measure_name[itype]
            for d, itype in self._extra_domain_integral_type_map.items()
        }
        return (
            f"{{self._integrand}} * "
            f"{mname}({self._ufl_domain}[{self._subdomain_id}], {self._metadata}, {temp})"
        )

    def __repr__(self):
        """Representation."""
        return (
            f"Integral({self._integrand!r}, {self._integral_type!r}, {self._ufl_domain!r}, "
            f"{self._subdomain_id!r}, {self._metadata!r}, {self._subdomain_data!r}, "
            f"extra_domain_integral_type_map={self._extra_domain_integral_type_map!r})"
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
            and self._extra_domain_integral_type_map == other._extra_domain_integral_type_map
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
            tuple((hash(d), itype) for d, itype in self._extra_domain_integral_type_map.items()),
        )
        return hash(hashdata)
