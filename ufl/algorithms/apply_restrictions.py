"""Apply restrictions.

This module contains the apply_restrictions algorithm which propagates
restrictions in a form towards the terminals.
"""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from typing import Union

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import Expr, Restricted
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import MeshSequence, extract_unique_domain
from ufl.sobolevspace import H1

default_restriction_map = {
    "cell": None,
    "exterior_facet": None,
    "exterior_facet_top": None,
    "exterior_facet_bottom": None,
    "exterior_facet_vert": None,
    "interior_facet": "+",
    "interior_facet_horiz": "+",
    "interior_facet_vert": "+",
}


class RestrictionPropagator(MultiFunction):
    """Restriction propagator."""

    def __init__(self, side=None, default_restrictions=None):
        """Initialise."""
        MultiFunction.__init__(self)
        self.current_restriction = side
        self.default_restrictions = default_restrictions
        # Caches for propagating the restriction with map_expr_dag
        self.vcaches = {"+": {}, "-": {}}
        self.rcaches = {"+": {}, "-": {}}
        if self.current_restriction is None:
            self._rp = {
                "+": RestrictionPropagator("+", default_restrictions),
                "-": RestrictionPropagator("-", default_restrictions),
            }

    def restricted(self, o):
        """When hitting a restricted quantity, visit child with a separate restriction algorithm."""
        # Assure that we have only two levels here, inside or outside
        # the Restricted node
        if self.current_restriction is not None:
            raise ValueError("Cannot restrict an expression twice.")
        # Configure a propagator for this side and apply to subtree
        side = o.side()
        return map_expr_dag(
            self._rp[side], o.ufl_operands[0], vcache=self.vcaches[side], rcache=self.rcaches[side]
        )

    # --- Reusable rules

    def _extract_and_check_domain(self, o):
        domain = extract_unique_domain(o, expand_mesh_sequence=False)
        if isinstance(domain, MeshSequence):
            try:
                (domain,) = set(domain.meshes)
            except ValueError:
                raise RuntimeError(
                    f"Not expecting a MeshSequence composed of "
                    f"multiple domains at this stage: found {domain}"
                )
        if domain not in self.default_restrictions:
            raise RuntimeError(f"Integral type on {domain} not known")
        return domain

    def _ignore_restriction(self, o):
        """Ignore current restriction.

        Quantity is independent of side also from a computational point
        of view.
        """
        return o

    def _require_restriction(self, o):
        """Restrict a discontinuous quantity to current side, require a side to be set."""
        if self.default_restrictions is None:
            # Just propagate restrictions.
            if self.current_restriction is None:
                return o
            else:
                return o(self.current_restriction)
        else:
            # Propagate restriction while checking validity.
            domain = self._extract_and_check_domain(o)
            r = self.default_restrictions[domain]
            if self.current_restriction is None:
                if r is None:
                    return o
                else:
                    raise ValueError(
                        f"Discontinuous type {o._ufl_class_.__name__} must be restricted."
                    )
            elif self.current_restriction in ["+", "-"]:
                if r not in ["+", "-"]:
                    raise ValueError(
                        f"Inconsistent restrictions: "
                        f"current restriction = {self.current_restriction}, while "
                        f"default restriction = {r}"
                    )
                return o(self.current_restriction)
            else:
                raise ValueError(f"Unknown restriction: {self.current_restriction}")

    def _default_restricted(self, o):
        """Restrict a continuous quantity to default side if no current restriction is set."""
        if self.default_restrictions is None:
            # Just propagate restrictions.
            if self.current_restriction is None:
                return o
            else:
                return o(self.current_restriction)
        else:
            # Propagate restriction while applying default.
            domain = self._extract_and_check_domain(o)
            r = self.default_restrictions[domain]
            if self.current_restriction is None:
                if r is None:
                    return o
                elif r in ["+", "-"]:
                    return o(r)
                else:
                    raise RuntimeError(f"Unknown default restriction {r} on domain {domain}")
            elif self.current_restriction in ["+", "-"]:
                if r not in ["+", "-"]:
                    raise ValueError(
                        f"Inconsistent restrictions: "
                        f"current restriction = {self.current_restriction}, while "
                        f"default restriction = {r}"
                    )
                return o(self.current_restriction)
            else:
                raise ValueError(f"Unknown restriction: {self.current_restriction}")

    def _opposite(self, o):
        """Restrict a quantity to default side.

        If the current restriction is different swap the sign, require a side to be set.
        """
        if self.default_restrictions is None:
            # Just propagate restrictions.
            if self.current_restriction is None:
                return o
            else:
                return o(self.current_restriction)
        else:
            domain = self._extract_and_check_domain(o)
            r = self.default_restrictions[domain]
            if self.current_restriction is None:
                if r is None:
                    return o
                else:
                    raise ValueError(
                        f"Discontinuous type {o._ufl_class_.__name__} must be restricted."
                    )
            elif self.current_restriction in ["+", "-"]:
                if r is None:
                    raise ValueError(
                        f"Inconsistent restrictions: "
                        f"current restriction = {self.current_restriction}, while "
                        f"default restriction = {r}"
                    )
                else:
                    if self.current_restriction == r:
                        return o(r)
                    else:
                        return -o(r)
            else:
                raise ValueError(f"Unknown restriction: {self.current_restriction}")

    def _missing_rule(self, o):
        """Raise an error."""
        raise ValueError(f"Missing rule for {o._ufl_class_.__name__}")

    # --- Rules for operators

    # Default: Operators should reconstruct only if subtrees are not touched
    operator = MultiFunction.reuse_if_untouched

    # Assuming apply_derivatives has been called,
    # propagating Grad inside the Restricted nodes.
    # Considering all grads to be discontinuous, may
    # want something else for facet functions in future.
    grad = _require_restriction

    def variable(self, o, op, label):
        """Strip variable."""
        return op

    def reference_value(self, o):
        """Reference value of something follows same restriction rule as the underlying object."""
        (f,) = o.ufl_operands
        assert f._ufl_is_terminal_
        g = self(f)
        if isinstance(g, Restricted):
            side = g.side()
            return o(side)
        else:
            return o

    # --- Rules for terminals

    # Require handlers to be specified for all terminals
    terminal = _missing_rule

    multi_index = _ignore_restriction
    label = _ignore_restriction

    # Default: Literals should ignore restriction
    constant_value = _ignore_restriction
    constant = _ignore_restriction

    # Even arguments with continuous elements such as Lagrange must be
    # restricted to associate with the right part of the element
    # matrix
    argument = _require_restriction

    # Defaults for geometric quantities
    geometric_cell_quantity = _require_restriction
    geometric_facet_quantity = _require_restriction

    # Only a few geometric quantities are independent on the restriction:
    facet_coordinate = _ignore_restriction
    quadrature_weight = _ignore_restriction

    # Assuming homogeoneous mesh
    reference_cell_volume = _ignore_restriction
    reference_facet_volume = _ignore_restriction

    # These are the same from either side but to compute them
    # cell (or facet) data from one side must be selected:
    spatial_coordinate = _default_restricted
    # Depends on cell only to get to the facet:
    facet_jacobian = _default_restricted
    facet_jacobian_determinant = _default_restricted
    facet_jacobian_inverse = _default_restricted
    facet_area = _default_restricted
    min_facet_edge_length = _default_restricted
    max_facet_edge_length = _default_restricted
    facet_origin = _default_restricted  # FIXME: Is this valid for quads?

    def coefficient(self, o):
        """Restrict a coefficient.

        Allow coefficients to be unrestricted (apply default if so) if
        the values are fully continuous across the facet.
        """
        if o.ufl_element() in H1:
            # If the coefficient _value_ is _fully_ continuous
            # It must still be computed from one of the sides, we just don't care which
            return self._default_restricted(o)
        else:
            return self._require_restriction(o)

    def facet_normal(self, o):
        """Restrict a facet_normal."""
        D = extract_unique_domain(o)
        e = D.ufl_coordinate_element()
        gd = D.geometric_dimension()
        td = D.topological_dimension()

        if e.embedded_superdegree <= 1 and e in H1 and gd == td:
            # For meshes with a continuous linear non-manifold
            # coordinate field, the facet normal from side - points in
            # the opposite direction of the one from side +.  We must
            # still require a side to be chosen by the user but
            # rewrite n- -> n+.  This is an optimization, possibly
            # premature, however it's more difficult to do at a later
            # stage.
            return self._opposite(o)
        else:
            # For other meshes, we require a side to be
            # chosen by the user and respect that
            return self._require_restriction(o)


def apply_restrictions(expression: Expr, default_restrictions: Union[dict, None] = None) -> Expr:
    """Propagate restriction nodes to wrap differential terminals directly.

    Args:
        expression: UFL expression.
        default_restrictions: domain-default_restriction map.
            If `None`, just propagate restrictions without
            applying the default restrictions.

    Returns:
        expression with the restriction nodes propagated.

    """
    rules = RestrictionPropagator(default_restrictions=default_restrictions)
    return map_integrand_dags(rules, expression)
