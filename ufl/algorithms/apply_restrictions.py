"""Apply restrictions.

This module contains the apply_restrictions algorithm which propagates restrictions in a form
towards the terminals.
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
from ufl.domain import extract_unique_domain, MixedMesh
from ufl.measure import integral_type_to_measure_name
from ufl.sobolevspace import H1
from ufl.classes import ReferenceGrad, ReferenceValue
from ufl.restriction import PositiveRestricted, SingleValueRestricted


class RestrictionPropagator(MultiFunction):
    """Restriction propagator."""

    def __init__(self, side=None, assume_single_integral_type=True):
        """Initialise."""
        MultiFunction.__init__(self)
        self.current_restriction = side
        self.default_restriction = "+" if assume_single_integral_type else "?"
        # Caches for propagating the restriction with map_expr_dag
        self.vcaches = {"+": {}, "-": {}, "|": {}, "?": {}}
        self.rcaches = {"+": {}, "-": {}, "|": {}, "?": {}}
        if self.current_restriction is None:
            self._rp = {"+": RestrictionPropagator("+", assume_single_integral_type),
                        "-": RestrictionPropagator("-", assume_single_integral_type),
                        "|": RestrictionPropagator("|", assume_single_integral_type),
                        "?": RestrictionPropagator("?", assume_single_integral_type)}
        self.assume_single_integral_type = assume_single_integral_type

    def restricted(self, o):
        """When hitting a restricted quantity, visit child with a separate restriction algorithm."""
        # Assure that we have only two levels here, inside or outside
        # the Restricted node
        if self.current_restriction is not None:
            raise ValueError("Cannot restrict an expression twice.")
        # Configure a propagator for this side and apply to subtree
        side = o.side()
        return map_expr_dag(self._rp[side], o.ufl_operands[0],
                            vcache=self.vcaches[side],
                            rcache=self.rcaches[side])

    # --- Reusable rules

    def _ignore_restriction(self, o):
        """Ignore current restriction, quantity is independent of side also from a computational point of view."""
        return o

    def _require_restriction(self, o):
        """Restrict a discontinuous quantity to current side, require a side to be set."""
        if self.current_restriction is not None:
            return o(self.current_restriction)
        elif not self.assume_single_integral_type:
            return o
        else:
            raise ValueError(f"Discontinuous type {o._ufl_class_.__name__} must be restricted.")

    def _default_restricted(self, o):
        """Restrict a continuous quantity to default side if no current restriction is set."""
        r = self.current_restriction
        if r is None:
            r = self.default_restriction
        return o(r)

    def _opposite(self, o):
        """Restrict a quantity to default side.

        If the current restriction is different swap the sign, require a side to be set.
        """
        if self.current_restriction is None:
            raise ValueError(f"Discontinuous type {o._ufl_class_.__name__} must be restricted.")
        elif self.current_restriction == self.default_restriction:
            return o(self.default_restriction)
        else:
            return -o(self.default_restriction)

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
        f, = o.ufl_operands
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

    def coefficient(self, o):
        """Restrict a coefficient.

        Allow coefficients to be unrestricted (apply default if so) if the values are fully continuous
        across the facet.
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


def apply_restrictions(expression, assume_single_integral_type=True):
    """Propagate restriction nodes to wrap differential terminals directly."""
    if assume_single_integral_type:
        integral_types = [k for k in integral_type_to_measure_name.keys()
                          if k.startswith("interior_facet")]
    else:
        # Integration type of the integral is not necessarily the same as
        # the integral type of a given function; e.g., the former can be
        # ``exterior_facet`` and the latter ``interior_facet``.
        integral_types = None
    rules = RestrictionPropagator(assume_single_integral_type=assume_single_integral_type)
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)


class DefaultRestrictionApplier(MultiFunction):
    """Default restriction applier."""

    def __init__(self, side=None, assume_single_integral_type=True):
        """Initialise."""
        MultiFunction.__init__(self)
        self.current_restriction = side
        # If multiple domains exist, the restriction on a function defined on
        # a certain domain can not be determined by merely inspecting the
        # local part of the DAG. "?" restrictions will be replaced with the
        # appropriate ones later using ``replace_to_be_restricted`` function.
        self.default_restriction = "+" if assume_single_integral_type else "?"

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


def apply_default_restrictions(expression, assume_single_integral_type=True):
    """Some terminals can be restricted from either side.

    This applies a default restriction to such terminals if unrestricted.
    """
    if assume_single_integral_type:
        integral_types = [k for k in integral_type_to_measure_name.keys()
                          if k.startswith("interior_facet")]
    else:
        integral_types = None
    rules = DefaultRestrictionApplier(assume_single_integral_type=assume_single_integral_type)
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)


class DomainRestrictionMapMaker(MultiFunction):
    """Make a map from domains to restriction(s).

    Inspect the DAG and collect domain-restrictions map.
    This must be done per integral_data.
    """

    def __init__(self, domain_restriction_map):
        MultiFunction.__init__(self)
        self._domain_restriction_map = domain_restriction_map

    expr = MultiFunction.reuse_if_untouched

    def _modifier(self, o):
        restriction = None
        local_derivatives = 0
        reference_value = False
        t = o
        while not t._ufl_is_terminal_:
            assert t._ufl_is_terminal_modifier_, f"Expecting a terminal modifier: got {repr(t)}"
            if isinstance(t, ReferenceValue):
                assert not reference_value, "Got twice pulled back terminal"
                reference_value = True
                t, = t.ufl_operands
            elif isinstance(t, ReferenceGrad):
                local_derivatives += 1
                t, = t.ufl_operands
            elif isinstance(t, Restricted):
                assert restriction is None, "Got twice restricted terminal"
                restriction = t._side
                t, = t.ufl_operands
            elif t._ufl_terminal_modifiers_:
                raise ValueError("Missing handler for terminal modifier type %s, object is %s." % (type(t), repr(t)))
            else:
                raise ValueError("Unexpected type %s object %s." % (type(t), repr(t)))
        domain = extract_unique_domain(t, expand_mixed_mesh=False)
        if isinstance(domain, MixedMesh):
            raise RuntimeError(f"Not expecting a terminal object on a mixed mesh at this stage: found {repr(t)}")
        if domain is not None:
            if domain not in self._domain_restriction_map:
                self._domain_restriction_map[domain] = set()
            if restriction in ['+', '-', '|']:
                self._domain_restriction_map[domain].add(restriction)
            elif restriction not in ['?', None]:
                raise RuntimeError
        return o

    reference_value = _modifier
    reference_grad = _modifier
    positive_restricted = _modifier
    negative_restricted = _modifier
    single_value_restricted = _modifier
    to_be_restricted = _modifier
    terminal = _modifier


def make_domain_restriction_map(integral_data):
    """Make domain-restriction map for the given integral_data."""
    domain_restriction_map = {}
    rule = DomainRestrictionMapMaker(domain_restriction_map)
    for integral in integral_data.integrals:
        _ = map_expr_dag(rule, integral.integrand())
    return domain_restriction_map


def make_domain_integral_type_map(integral_data):
    domain_restriction_map = make_domain_restriction_map(integral_data)
    integration_domain = integral_data.domain
    integration_type = integral_data.integral_type
    domain_integral_type_dict = {}
    for d, rs in domain_restriction_map.items():
        if rs in [{'+'}, {'-'}, {'+', '-'}]:
            domain_integral_type_dict[d] = "interior_facet"
        elif rs == {'|'}:
            domain_integral_type_dict[d] = "exterior_facet"
        elif rs == set():
            if d.topological_dimension() == integration_domain.topological_dimension():
                if integration_type == "cell":
                    domain_integral_type_dict[d] = "cell"
                elif integration_type in ["exterior_facet", "interior_facet"]:
                    domain_integral_type_dict[d] = "exterior_facet"
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise RuntimeError(f"Found inconsistent restrictions {rs} for domain {d}")
    if integration_domain in domain_integral_type_dict:
        if domain_integral_type_dict[integration_domain] != integration_type:
            raise RuntimeError(f"""Found inconsistent integral types for the integration domain ({integration_domain}) :
                {domain_integral_type_dict[integration_domain]} != {integration_type}""")
    else:
        domain_integral_type_dict[integration_domain] = integration_type
    return domain_integral_type_dict


class ToBeRestrectedReplacer(MultiFunction):
    """Replace ``?`` restrictions."""

    def __init__(self, domain_integral_type_map):
        MultiFunction.__init__(self)
        self.domain_integral_type_map = domain_integral_type_map

    expr = MultiFunction.reuse_if_untouched

    def to_be_restricted(self, o):
        mt, = o.ufl_operands
        domain = extract_unique_domain(mt)
        if isinstance(domain, MixedMesh):
            raise RuntimeError(f"""Not expecting a (modified) terminal object on a mixed mesh at this stage :
                got {repr(o)}""")
        if domain not in self.domain_integral_type_map:
            raise RuntimeError(f"Integral type on {domain} not known")
        integral_type = self.domain_integral_type_map[domain]
        if integral_type == "cell":
            return mt
        elif integral_type == "exterior_facet":
            return SingleValueRestricted(mt)
        elif integral_type == "interial_facet":
            return PositiveRestricted(mt)
        else:
            raise RuntimeError(f"Unknown integral type: {integral_type}")


def replace_to_be_restricted(integral_data):
    new_integrals = []
    rule = ToBeRestrectedReplacer(integral_data.domain_integral_type_map)
    for integral in integral_data.integrals:
        integrand = map_expr_dag(rule, integral.integrand())
        new_integrals.append(integral.reconstruct(integrand=integrand))
    return new_integrals
