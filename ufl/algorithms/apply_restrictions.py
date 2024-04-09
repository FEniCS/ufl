"""Apply restrictions.

This module contains the apply_restrictions algorithm which propagates restrictions in a form
towards the terminals.
"""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections import defaultdict
import numpy

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import Restricted
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_unique_domain, MixedMesh
from ufl.measure import integral_type_to_measure_name
from ufl.sobolevspace import H1
from ufl.geometry import GeometricQuantity
from ufl.classes import Form, ReferenceGrad, ReferenceValue, Restricted, Indexed, MultiIndex, Index, FixedIndex, ComponentTensor, ListTensor, Zero
from ufl.coefficient import Coefficient
from ufl import indices
from ufl.checks import is_cellwise_constant
from ufl.tensors import as_tensor
from ufl.restriction import NegativeRestricted, PositiveRestricted, SingleValueRestricted, ToBeRestricted


class RestrictionPropagator(MultiFunction):
    """Restriction propagator."""

    def __init__(self, side=None, have_multiple_domains=False):
        """Initialise."""
        MultiFunction.__init__(self)
        self.current_restriction = side
        self.default_restriction = "?" if have_multiple_domains else "+"
        # Caches for propagating the restriction with map_expr_dag
        self.vcaches = {"+": {}, "-": {}, "|": {}, "?": {}}
        self.rcaches = {"+": {}, "-": {}, "|": {}, "?": {}}
        if self.current_restriction is None:
            self._rp = {"+": RestrictionPropagator("+", have_multiple_domains),
                        "-": RestrictionPropagator("-", have_multiple_domains),
                        "|": RestrictionPropagator("|", have_multiple_domains),
                        "?": RestrictionPropagator("?", have_multiple_domains)}
        self.have_multiple_domains = have_multiple_domains

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
        elif self.have_multiple_domains:
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


def apply_restrictions(expression, have_multiple_domains=False):
    """Propagate restriction nodes to wrap differential terminals directly."""
    if have_multiple_domains:
        integral_types = None
    else:
        integral_types = [k for k in integral_type_to_measure_name.keys()
                          if k.startswith("interior_facet")]
    rules = RestrictionPropagator(have_multiple_domains=have_multiple_domains)
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)


class DefaultRestrictionApplier(MultiFunction):
    """Default restriction applier."""

    def __init__(self, side=None, have_multiple_domains=False):
        """Initialise."""
        MultiFunction.__init__(self)
        self.current_restriction = side
        self.default_restriction = "?" if have_multiple_domains else "+"
        if self.current_restriction is None:
            self._rp = {"+": DefaultRestrictionApplier("+", have_multiple_domains),
                        "-": DefaultRestrictionApplier("-", have_multiple_domains),
                        "|": DefaultRestrictionApplier("|", have_multiple_domains),
                        "?": DefaultRestrictionApplier("?", have_multiple_domains)}
        self.have_multiple_domains = have_multiple_domains

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


def apply_default_restrictions(expression, have_multiple_domains=False):
    """Some terminals can be restricted from either side.

    This applies a default restriction to such terminals if unrestricted.
    """
    if have_multiple_domains:
        integral_types = None
    else:
        integral_types = [k for k in integral_type_to_measure_name.keys()
                          if k.startswith("interior_facet")]
    rules = DefaultRestrictionApplier(have_multiple_domains=have_multiple_domains)
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)


class DomainRestrictionMapMaker(MultiFunction):
    """DomainRestrictionMapMaker."""

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
            assert t._ufl_is_terminal_modifier_, f"Got {repr(t)}"
            if isinstance(t, ReferenceValue):
                assert not reference_value, "Got twice pulled back terminal!"
                reference_value = True
                t, = t.ufl_operands
            elif isinstance(t, ReferenceGrad):
                local_derivatives += 1
                t, = t.ufl_operands
            elif isinstance(t, Restricted):
                assert restriction is None, "Got twice restricted terminal!"
                restriction = t._side
                t, = t.ufl_operands
            elif t._ufl_terminal_modifiers_:
                raise ValueError("Missing handler for terminal modifier type %s, object is %s." % (type(t), repr(t)))
            else:
                raise ValueError("Unexpected type %s object %s." % (type(t), repr(t)))
        domain = extract_unique_domain(t, expand_mixed_mesh=False)
        if isinstance(domain, MixedMesh):
            raise RuntimeError(f"Not expecting a mixed mesh at this stage: {repr(t)}")
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
    """Collect domain_restriction map."""
    domain_restriction_map = {}
    rule = DomainRestrictionMapMaker(domain_restriction_map)
    for integral in integral_data.integrals:
        integrand = integral.integrand()
        _ = map_expr_dag(rule, integrand)
    return domain_restriction_map


def make_domain_integral_type_map(domain_restriction_map, integration_domain, integration_type):
    # Have no mixed mesh support
    #if integration_type in ["exterior_facet_top", "exterior_facet_bottom", "exterior_facet_vert", "interior_facet_vert", "interior_facet_horiz"]:
    #    return {integration_domain: integration_type}, [False for _ in integral_data.integrals]
    #domain_restrictions_map, integrand_is_zero = make_domain_restrictions_map(integral_data, form_data)
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
            raise RuntimeError(f"domain ({integration_domain}) has inconsistent restrictions : {domain_integral_type_dict[integration_domain]} != {integration_type}")
    else:
        domain_integral_type_dict[integration_domain] = integration_type
    return domain_integral_type_dict


# apply_coefficient_split


class CoefficientSplitter(MultiFunction):

    def __init__(self, coefficient_split):
        MultiFunction.__init__(self)
        self._coefficient_split = coefficient_split

    expr = MultiFunction.reuse_if_untouched

    def modified_terminal(self, o):
        restriction = None
        local_derivatives = 0
        reference_value = False
        t = o
        while not t._ufl_is_terminal_:
            print("")
            print(repr(t))
            assert t._ufl_is_terminal_modifier_, f"Got {repr(t)}"
            if isinstance(t, ReferenceValue):
                assert not reference_value, "Got twice pulled back terminal!"
                reference_value = True
                t, = t.ufl_operands
            elif isinstance(t, ReferenceGrad):
                local_derivatives += 1
                t, = t.ufl_operands
            elif isinstance(t, Restricted):
                assert restriction is None, "Got twice restricted terminal!"
                restriction = t._side
                t, = t.ufl_operands
            elif t._ufl_terminal_modifiers_:
                raise ValueError("Missing handler for terminal modifier type %s, object is %s." % (type(t), repr(t)))
            else:
                raise ValueError("Unexpected type %s object %s." % (type(t), repr(t)))
        if not isinstance(t, Coefficient):
            # Only split coefficients
            return o
        if t not in self._coefficient_split:
            # Only split mixed coefficients
            return o
        # Reference value expected
        assert reference_value
        # Derivative indices
        beta = indices(local_derivatives)
        components = []
        for subcoeff in self._coefficient_split[t]:
            c = subcoeff
            # Apply terminal modifiers onto the subcoefficient
            if reference_value:
                c = ReferenceValue(c)
            for n in range(local_derivatives):
                # Return zero if expression is trivially constant. This has to
                # happen here because ReferenceGrad has no access to the
                # topological dimension of a literal zero.
                if is_cellwise_constant(c):
                    dim = extract_unique_domain(subcoeff).topological_dimension()
                    c = Zero(c.ufl_shape + (dim,), c.ufl_free_indices, c.ufl_index_dimensions)
                else:
                    c = ReferenceGrad(c)
            if restriction == '+':
                c = PositiveRestricted(c)
            elif restriction == '-':
                c = NegativeRestricted(c)
            elif restriction == '|':
                c = SingleValueRestricted(c)
            elif restriction == '?':
                c = ToBeRestricted(c)
            elif restriction is not None:
                raise RuntimeError(f"Got unknown restriction: {restriction}")
            # Collect components of the subcoefficient
            for alpha in numpy.ndindex(subcoeff.ufl_element().reference_value_shape):
                # New modified terminal: component[alpha + beta]
                components.append(c[alpha + beta])
        # Repack derivative indices to shape
        c, = indices(1)
        return ComponentTensor(as_tensor(components)[c], MultiIndex((c,) + beta))

    positive_restricted = modified_terminal
    negative_restricted = modified_terminal
    single_value_restricted = modified_terminal
    to_be_restricted = modified_terminal
    reference_grad = modified_terminal
    reference_value = modified_terminal
    terminal = modified_terminal


def apply_coefficient_split(expr, coefficient_split):
    """Split mixed coefficients, so mixed elements need not be
    implemented.

    :arg split: A :py:class:`dict` mapping each mixed coefficient to a
                sequence of subcoefficients.  If None, calling this
                function is a no-op.
    """
    if coefficient_split is None:
        return expr
    splitter = CoefficientSplitter(coefficient_split)
    return map_expr_dag(splitter, expr)


class FixedIndexRemover(MultiFunction):

    def __init__(self, fimap):
        MultiFunction.__init__(self)
        self.fimap = fimap

    expr = MultiFunction.reuse_if_untouched

    def zero(self, o):
        free_indices = []
        index_dimensions = []
        for i, d in zip(o.ufl_free_indices, o.ufl_index_dimensions):
            if Index(i) in self.fimap:
                ind_j = self.fimap[Index(i)]
                if not isinstance(ind_j, FixedIndex):
                    free_indices.append(ind_j.count())
                    index_dimensions.append(d)
            else:
                free_indices.append(i)
                index_dimensions.append(d)
        return Zero(shape=o.ufl_shape, free_indices=tuple(free_indices), index_dimensions=tuple(index_dimensions))

    def list_tensor(self, o):
        rule = FixedIndexRemover(self.fimap)
        cc = []
        for o1 in o.ufl_operands:
            comp = map_expr_dag(rule, o1)
            cc.append(comp)
        return ListTensor(*cc)

    def multi_index(self, o):
        return MultiIndex(tuple(self.fimap.get(i, i) for i in o.indices()))


class IndexRemover(MultiFunction):

    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def _zero_simplify(self, o):
        operand, = o.ufl_operands
        rule = IndexRemover()
        operand = map_expr_dag(rule, operand)
        if isinstance(operand, Zero):
            return Zero(shape=o.ufl_shape, free_indices=o.ufl_free_indices, index_dimensions=o.ufl_index_dimensions)
        else:
            return o._ufl_expr_reconstruct_(operand)

    def indexed(self, o):
        o1, i1 = o.ufl_operands
        if isinstance(o1, ComponentTensor):
            o2, i2 = o1.ufl_operands
            fimap = dict(zip(i2.indices(), i1.indices(), strict=True))
            rule = FixedIndexRemover(fimap)
            v = map_expr_dag(rule, o2)
            rule = IndexRemover()
            return map_expr_dag(rule, v)
        elif isinstance(o1, ListTensor):
            if isinstance(i1[0], FixedIndex):
                o1 = o1.ufl_operands[i1[0]._value]
                rule = IndexRemover()
                if len(i1) > 1:
                    i1 = MultiIndex(i1[1:])
                    return map_expr_dag(rule, Indexed(o1, i1))
                else:
                    return map_expr_dag(rule, o1)
        rule = IndexRemover()
        o1 = map_expr_dag(rule, o1)
        return Indexed(o1, i1)

    # Do something nicer
    positive_restricted = _zero_simplify
    negative_restricted = _zero_simplify
    single_value_restricted = _zero_simplify
    to_be_restricted = _zero_simplify
    reference_grad = _zero_simplify
    reference_value = _zero_simplify


def remove_component_and_list_tensors(o):
    if isinstance(o, Form):
        integrals = []
        for integral in o.integrals():
            integrand = remove_component_list_tensors(integral.integrand())
            if not isinstance(integrand, Zero):
                integrals.append(integral.reconstruct(integrand=integrand))
        return o._ufl_expr_reconstruct_(integrals)
    else:
        rule = IndexRemover()
        return map_expr_dag(rule, o)
