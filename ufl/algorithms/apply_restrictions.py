"""Apply restrictions.

This module contains the apply_restrictions algorithm which propagates
restrictions in a form towards the terminals.
"""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import typing
from typing import Protocol

from ufl.algorithms.map_integrands import map_integrand_dags, map_integrand_dags_legacy
from ufl.core.ufl_type import UFLObject
from ufl.corealg.multifunction import MultiFunction
from ufl.measure import integral_type_to_measure_name
from ufl.typing import Self


class ApplyRestrictions(Protocol):
    """Protocol for apply_restrictions."""

    def apply_restrictions(
        self, mapped_operators: typing.Tuple[UFLObject, ...], side: typing.Optional[str]
    ) -> Self:
        """Apply restrictions.

        Propagates restrictions in a form towards the terminals.
        """


default_restriction = "+"


def apply_restrictions(expression):
    """Propagate restriction nodes to wrap differential terminals directly."""
    integral_types = [
        k for k in integral_type_to_measure_name.keys() if k.startswith("interior_facet")
    ]

    return map_integrand_dags(
        "apply_restrictions", (None,), expression, only_integral_type=integral_types, cutoff=True
    )


class DefaultRestrictionApplier(MultiFunction):
    """Default restriction applier."""

    def __init__(self, side=None):
        """Initialise."""
        MultiFunction.__init__(self)
        self.current_restriction = side
        self.default_restriction = "+"
        if self.current_restriction is None:
            self._rp = {"+": DefaultRestrictionApplier("+"), "-": DefaultRestrictionApplier("-")}

    def terminal(self, o):
        """Apply to terminal."""
        # Most terminals are unchanged
        return o

    # Default: Operators should reconstruct only if subtrees are not touched
    operator = MultiFunction.reuse_if_untouched

    def restricted(self, o):
        """Apply to restricted."""
        # Don't restrict twice
        return o

    def derivative(self, o):
        """Apply to derivative."""
        # I don't think it's safe to just apply default restriction
        # to the argument of any derivative, i.e. grad(cg1_function)
        # is not continuous across cells even if cg1_function is.
        return o

    def _default_restricted(self, o):
        """Restrict a continuous quantity to default side if no current restriction is set."""
        r = self.current_restriction
        if r is None:
            r = self.default_restriction
        return o(r)

    # These are the same from either side but to compute them
    # cell (or facet) data from one side must be selected:
    spatial_coordinate = _default_restricted
    # Depends on cell only to get to the facet:
    facet_jacobian = _default_restricted
    facet_jacobian_determinant = _default_restricted
    facet_jacobian_inverse = _default_restricted
    # facet_tangents = _default_restricted
    # facet_midpoint = _default_restricted
    facet_area = _default_restricted
    # facet_diameter = _default_restricted
    min_facet_edge_length = _default_restricted
    max_facet_edge_length = _default_restricted
    facet_origin = _default_restricted  # FIXME: Is this valid for quads?


def apply_default_restrictions(expression):
    """Some terminals can be restricted from either side.

    This applies a default restriction to such terminals if unrestricted.
    """
    integral_types = [
        k for k in integral_type_to_measure_name.keys() if k.startswith("interior_facet")
    ]
    rules = DefaultRestrictionApplier()
    return map_integrand_dags_legacy(rules, expression, only_integral_type=integral_types)
