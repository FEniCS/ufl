"""This module contains the apply_restrictions algorithm which propagates
restrictions in a form towards the terminals."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import (Restricted, Operator, Grad, Variable, ReferenceValue, MultiIndex, Label,
                        ConstantValue, Constant, Argument, Terminal, GeometricCellQuantity, GeometricFacetQuantity,
                        FacetCoordinate, QuadratureWeight, ReferenceCellVolume, ReferenceFacetVolume, Coefficient, FacetNormal,
                        Derivative, SpatialCoordinate, FacetJacobian, FacetJacobianDeterminant, FacetJacobianInverse, FacetArea, MinFacetEdgeLength,
                        MaxFacetEdgeLength, FacetOrigin, PositiveRestricted, NegativeRestricted)
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction, reuse_if_untouched
from ufl.domain import extract_unique_domain
from ufl.measure import integral_type_to_measure_name
from ufl.sobolevspace import H1
from functools import singledispatch

@singledispatch
def _restriction_propagator(o, self):
    """Single-dispatch function to propagate restriction through an expression

    :arg o: UFL expression
    :arg self: wrapper class that manages caches

    """
    raise AssertionError("UFL node expected, not %s" % type(o))


@_restriction_propagator.register(Restricted)
@_restriction_propagator.register(PositiveRestricted)
@_restriction_propagator.register(NegativeRestricted)
def _restriction_propagator_restricted(o, self, *args):
    "When hitting a restricted quantity, visit child with a separate restriction algorithm."
    # Assure that we have only two levels here, inside or outside
    # the Restricted node
    if self.current_restriction is not None:
        raise ValueError("Cannot restrict an expression twice.")
    # Configure a propagator for this side and apply to subtree
    side = o.side()
    print("RESTRICTED")
    return map_expr_dag(self._rp[side], o.ufl_operands[0],
                        vcache=self.vcaches[side],
                        rcache=self.rcaches[side])


# Default: Operators should reconstruct only if subtrees are not touched
@_restriction_propagator.register(Operator)
def _restriction_propagator_default(o, self, *args):
    print("operator")
    return reuse_if_untouched(o, *args)


@_restriction_propagator.register(Variable)
def _restriction_propagator_variable(o, self, op, label):
    "Strip variable."
    return op


@_restriction_propagator.register(ReferenceValue)
def _restriction_propagator_reference_value(o, self):
    "Reference value of something follows same restriction rule as the underlying object."
    f, = o.ufl_operands
    assert f._ufl_is_terminal_
    g = self(f)
    if isinstance(g, Restricted):
        side = g.side()
        return o(side)
    else:
        return o


@_restriction_propagator.register(MultiIndex)
@_restriction_propagator.register(Label)
# Default: Literals should ignore restriction
@_restriction_propagator.register(ConstantValue)
@_restriction_propagator.register(Constant)
# Only a few geometric quantities are independent on the restriction:
@_restriction_propagator.register(FacetCoordinate)
@_restriction_propagator.register(QuadratureWeight)
# Assuming homogeoneous mesh
@_restriction_propagator.register(ReferenceCellVolume)
@_restriction_propagator.register(ReferenceFacetVolume)
def _ignore_restriction(o, self):
    "Ignore current restriction, quantity is independent of side also from a computational point of view."
    return o


# Assuming apply_derivatives has been called,
# propagating Grad inside the Restricted nodes.
# Considering all grads to be discontinuous, may
# want something else for facet functions in future.
@_restriction_propagator.register(Grad)
# Even arguments with continuous elements such as Lagrange must be
# restricted to associate with the right part of the element
# matrix
@_restriction_propagator.register(Argument)
# Defaults for geometric quantities
@_restriction_propagator.register(GeometricCellQuantity)
@_restriction_propagator.register(GeometricFacetQuantity)
def _require_restriction(o, self):
    "Restrict a discontinuous quantity to current side, require a side to be set."
    print(type(o))
    print(self)
    if self.current_restriction is None:
        raise ValueError(f"Discontinuous type {o._ufl_class_.__name__} must be restricted.")
    return o(self.current_restriction)


def _opposite(o, self):
    """Restrict a quantity to default side, if the current restriction
    is different swap the sign, require a side to be set."""
    if self.current_restriction is None:
        raise ValueError(f"Discontinuous type {o._ufl_class_.__name__} must be restricted.")
    elif self.current_restriction == self.default_restriction:
        return o(self.default_restriction)
    else:
        return -o(self.default_restriction)


@_restriction_propagator.register(Terminal)
def _missing_rule(o, self):
    raise ValueError(f"Missing rule for {o._ufl_class_.__name__}")


@_restriction_propagator.register(Coefficient)
def _restriction_propagator_coefficient(o, self):
        """Allow coefficients to be unrestricted (apply default if so) if the values are
        fully continuous across the facet."""
        print("Coefficient")
        print(o.ufl_element() in H1)
        print(self.current_restriction)
        if o.ufl_element() in H1:
            # If the coefficient _value_ is _fully_ continuous
            return _default_restricted(o, self)  # Must still be computed from one of the sides, we just don't care which
        else:
            return _require_restriction(o, self)
        

@_restriction_propagator.register(FacetNormal)
def _restriction_propagator_facet_normal(o, self):
    D = extract_unique_domain(o)
    e = D.ufl_coordinate_element()
    gd = D.geometric_dimension()
    td = D.topological_dimension()

    if e._is_linear() and gd == td:
        # For meshes with a continuous linear non-manifold
        # coordinate field, the facet normal from side - points in
        # the opposite direction of the one from side +.  We must
        # still require a side to be chosen by the user but
        # rewrite n- -> n+.  This is an optimization, possibly
        # premature, however it's more difficult to do at a later
        # stage.
        return _opposite(o, self)
    else:
        # For other meshes, we require a side to be
        # chosen by the user and respect that
        return _require_restriction(o, self)



class RestrictionPropagator(object):
    def __init__(self, side=None):
        self.function = _restriction_propagator
        self.current_restriction = side
        self.default_restriction = "+"
        # Caches for propagating the restriction with map_expr_dag
        self.vcaches = {"+": {}, "-": {}}
        self.rcaches = {"+": {}, "-": {}}
        if self.current_restriction is None:
            self._rp = {"+": RestrictionPropagator("+"),
                        "-": RestrictionPropagator("-")}
            
        
    def __call__(self, node, *args):
        return self.function(node, self, *args)


def apply_restrictions(expression):
    "Propagate restriction nodes to wrap differential terminals directly."
    integral_types = [k for k in integral_type_to_measure_name.keys()
                      if k.startswith("interior_facet")]
    rules = RestrictionPropagator()
    print("applying restriction")
    print(type(expression))
    if isinstance(expression, Restricted):
        print("restricted")
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)


class DefaultRestrictionApplier(MultiFunction):
    def __init__(self, side=None):
        self.function = _default_restriction_applier
        self.current_restriction = side
        self.default_restriction = "+"
        if self.current_restriction is None:
            self._rp = {"+": DefaultRestrictionApplier("+"),
                        "-": DefaultRestrictionApplier("-")}
            
    def __call__(self, node, *args):
        return self.function(node, self, *args)


@singledispatch
def _default_restriction_applier(o, self):
    """Single-dispatch function to apply restriction to an expression

    :arg o: UFL expression
    :arg self: wrapper class that manages caches

    """
    raise AssertionError("UFL node expected, not %s" % type(o))
   
@_default_restriction_applier.register(Terminal)
def _default_restriction_applier_terminal(o, self):
    # Most terminals are unchanged
    return o

# Default: Operators should reconstruct only if subtrees are not touched
_default_restriction_applier.register(Operator)
def _default_restriction_applier_default(o, self, *args):
    print("operator")
    return reuse_if_untouched(o, *args)

_default_restriction_applier.register(Restricted)
def _default_restriction_applier_restricted(o, self):
        # Don't restrict twice
        return o

_default_restriction_applier.register(Derivative)
def _default_restriction_applier_derivative(o, self):
        # I don't think it's safe to just apply default restriction
        # to the argument of any derivative, i.e. grad(cg1_function)
        # is not continuous across cells even if cg1_function is.
        return o

# These are the same from either side but to compute them
# cell (or facet) data from one side must be selected:
_default_restriction_applier.register(SpatialCoordinate)
# Depends on cell only to get to the facet:
_default_restriction_applier.register(FacetJacobian)
_default_restriction_applier.register(FacetJacobianDeterminant)
_default_restriction_applier.register(FacetJacobianInverse)
_default_restriction_applier.register(FacetArea)
_default_restriction_applier.register(MinFacetEdgeLength)
_default_restriction_applier.register(MaxFacetEdgeLength)
_default_restriction_applier.register(FacetOrigin) # FIXME: Is this valid for quads?
def _default_restricted(o, self):
    "Restrict a continuous quantity to default side if no current restriction is set."
    r = self.current_restriction
    if r is None:
        r = self.default_restriction
    return o(r)


def apply_default_restrictions(expression):
    """Some terminals can be restricted from either side.

    This applies a default restriction to such terminals if unrestricted."""
    integral_types = [k for k in integral_type_to_measure_name.keys()
                      if k.startswith("interior_facet")]
    rules = DefaultRestrictionApplier()
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)
