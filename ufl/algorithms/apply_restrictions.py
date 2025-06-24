"""Apply restrictions.

This module contains the apply_restrictions algorithm which propagates
restrictions in a form towards the terminals.
"""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import Restricted
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import MeshSequence, extract_domains, extract_unique_domain
from ufl.measure import integral_type_to_measure_name
from ufl.sobolevspace import H1


class RestrictionPropagator(MultiFunction):
    """Restriction propagator."""

    def __init__(self, side=None, default_restriction=None):
        """Initialise."""
        MultiFunction.__init__(self)
        self.current_restriction = side
        self.default_restriction = default_restriction
        # Caches for propagating the restriction with map_expr_dag
        self.vcaches = {
            "+": {},
            "-": {},
        }
        self.rcaches = {
            "+": {},
            "-": {},
        }
        if self.current_restriction is None:
            self._rp = {
                "+": RestrictionPropagator("+", default_restriction),
                "-": RestrictionPropagator("-", default_restriction),
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

    def _ignore_restriction(self, o):
        """Ignore current restriction.

        Quantity is independent of side also from a computational point
        of view.
        """
        return o

    def _require_restriction(self, o):
        """Restrict a discontinuous quantity to current side, require a side to be set."""
        if self.current_restriction is not None:
            return o(self.current_restriction)
        if self.default_restriction is not None:
            domain = extract_unique_domain(o, expand_mesh_sequence=False)
            if isinstance(domain, MeshSequence):
                raise RuntimeError(
                    f"Not expecting a terminal object on a mixed mesh at this stage: found {o!r}"
                )
            if isinstance(self.default_restriction, dict):
                r = self.default_restriction[domain]
            else:
                r = self.default_restriction
            if r is None:
                # If integration if over interior facet of meshA and exterior facet of meshB,
                # arguments (say) on meshA must be restricted, but those on meshB do not
                # need to be.
                return o
            else:
                raise ValueError(f"Discontinuous type {o._ufl_class_.__name__} must be restricted.")
        else:
            return o

    def _default_restricted(self, o):
        """Restrict a continuous quantity to default side if no current restriction is set."""
        r = self.current_restriction
        if r is not None:
            return o(r)
        if self.default_restriction is not None:
            domain = extract_unique_domain(o, expand_mesh_sequence=False)
            if isinstance(domain, MeshSequence):
                raise RuntimeError(
                    f"Not expecting a terminal object on a mixed mesh at this stage: found {o!r}"
                )
            if isinstance(self.default_restriction, dict):
                if domain not in self.default_restriction:
                    raise RuntimeError(f"Integral type on {domain} not known")
                r = self.default_restriction[domain]
                if r is None:
                    return o
                elif r in ["+", "-"]:
                    return o(r)
                else:
                    raise RuntimeError(f"Unknown default restriction {r} on domain {domain}")
            else:
                # conventional "+" default:
                return o(self.default_restriction)
        else:
            return o

    def _opposite(self, o):
        """Restrict a quantity to default side.

        If the current restriction is different swap the sign, require a side to be set.
        """
        if isinstance(self.default_restriction, dict):
            domain = extract_unique_domain(o, expand_mesh_sequence=False)
            if isinstance(domain, MeshSequence):
                raise RuntimeError(
                    f"Not expecting a terminal object on a mixed mesh at this stage: found {o!r}"
                )
            if domain not in self.default_restriction:
                raise RuntimeError(f"Integral type on {domain} not known")
            r = self.default_restriction[domain]
        else:
            r = self.default_restriction
        if r is None:
            if self.current_restriction is not None:
                raise ValueError(
                    f"Expecting current_restriction None: got {self.current_restriction}"
                )
            return o
        else:
            if self.current_restriction is None:
                raise ValueError(f"Discontinuous type {o._ufl_class_.__name__} must be restricted.")
            elif self.current_restriction == r:
                return o(r)
            else:
                return -o(r)

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


def apply_restrictions(expression, assume_single_integral_type=True, domain_integral_type_map=None):
    """Propagate restriction nodes to wrap differential terminals directly."""
    if assume_single_integral_type:
        # Hnadle the conventional single-domain case.
        domains = extract_domains(expression)
        default_restriction = {domain: "+" for domain in domains}
        integral_types = [
            k for k in integral_type_to_measure_name.keys() if k.startswith("interior_facet")
        ]
    else:
        if domain_integral_type_map is None:
            # Do not apply default restrictions.
            default_restriction = None
        else:
            # Apply default restriction depending on the integral type on each domain.
            default_restriction = {
                domain: {
                    "cell": None,
                    "exterior_facet": None,
                    "exterior_facet_top": None,
                    "exterior_facet_bottom": None,
                    "exterior_facet_vert": None,
                    "interior_facet": "+",
                    "interior_facet_horiz": "+",
                    "interior_facet_vert": "+",
                }[integral_type]
                for domain, integral_type in domain_integral_type_map.items()
            }
        # Integration type of the integral is not necessarily the same as
        # the integral type of a given function; e.g., the former can be
        # ``exterior_facet`` and the latter ``interior_facet``.
        integral_types = None
    rules = RestrictionPropagator(default_restriction=default_restriction)
    return map_integrand_dags(rules, expression, only_integral_type=integral_types)
