"""Apply restrictions.

This module contains the apply_restrictions algorithm which propagates
restrictions in a form towards the terminals.
"""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

from functools import singledispatchmethod
from typing import Literal, cast

import ufl.classes
from ufl.algorithms.map_integrands import map_integrands
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import Mesh, extract_unique_domain
from ufl.integral import Integral
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


class RestrictionPropagator(DAGTraverser):
    """Restriction propagator."""

    def __init__(
        self,
        side: Literal["+", "-"] | None = None,
        default_restrictions: dict[Mesh, Literal["+", "-"] | None] | None = None,
    ):
        """Initialise a restriction propagator.

        Args:
            side: The side of the mesh to restrict to, if `None`, no restriction.
            default_restrictions: A map between meshes and certain restrictions
                set by the integration measure.
        """
        super().__init__()
        self.current_restriction: Literal["+", "-"] | None = side
        self.default_restrictions = default_restrictions
        if self.current_restriction is None:
            self._rp = {
                "+": RestrictionPropagator("+", default_restrictions),
                "-": RestrictionPropagator("-", default_restrictions),
            }

    @singledispatchmethod
    def process(self, o: ufl.classes.Expr):
        """Process an expression."""
        return super().process(o)

    @process.register(ufl.classes.Operator)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Operator, *operands):
        """Reconstruct an operator if restriction propagation changed an operand."""
        if all(new is old for new, old in zip(operands, o.ufl_operands)):
            return o
        return o._ufl_expr_reconstruct_(*operands)

    @process.register(ufl.classes.Restricted)
    def _(self, o: ufl.classes.Restricted):
        """When hitting a restricted quantity, visit child with a separate restriction algorithm."""
        # Assure that we have only two levels here, inside or outside
        # the Restricted node
        if self.current_restriction is not None:
            raise ValueError("Cannot restrict an expression twice.")
        # Configure a propagator for this side and apply to subtree
        side = o.side()
        return self._rp[side](o.ufl_operands[0])

    # --- Reusable rules

    def _extract_and_check_domain(self, o):
        """Extract single domain from a ufl."""
        domain = extract_unique_domain(o, expand_mesh_sequence=True)
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

    # Assuming apply_derivatives has been called,
    # propagating Grad inside the Restricted nodes.
    # Considering all grads to be discontinuous, may
    # want something else for facet functions in future.
    @process.register(ufl.classes.Grad)
    def _(self, o: ufl.classes.Grad):
        return self._require_restriction(o)

    @process.register(ufl.classes.Variable)
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Variable, op, label):
        """Strip variable."""
        return op

    @process.register(ufl.classes.ReferenceValue)
    def _(self, o: ufl.classes.ReferenceValue):
        """Reference value of something follows same restriction rule as the underlying object."""
        (f,) = o.ufl_operands
        assert f._ufl_is_terminal_ or isinstance(f, ufl.classes.Interpolate)
        g = self(f)
        if isinstance(g, ufl.classes.Restricted):
            side = g.side()
            return o(side)
        else:
            return o

    # --- Rules for terminals

    # Require handlers to be specified for all terminals
    @process.register(ufl.classes.Terminal)
    def _(self, o: ufl.classes.Terminal):
        return self._missing_rule(o)

    # These types are independent of restrictions.
    @process.register(ufl.classes.MultiIndex)
    @process.register(ufl.classes.Label)
    @process.register(ufl.classes.ConstantValue)
    @process.register(ufl.classes.Constant)
    @process.register(ufl.classes.FacetCoordinate)
    @process.register(ufl.classes.QuadratureWeight)
    @process.register(ufl.classes.ReferenceCellVolume)
    @process.register(ufl.classes.ReferenceFacetVolume)
    def _(self, o):
        return self._ignore_restriction(o)

    # Even arguments with continuous elements such as Lagrange must be
    # restricted to associate with the right part of the element
    # matrix
    @process.register(ufl.classes.Argument)
    @process.register(ufl.classes.GeometricCellQuantity)
    @process.register(ufl.classes.GeometricFacetQuantity)
    def _(self, o):
        return self._require_restriction(o)

    # These quantities are the same from either side but must be computed
    # from one side to get the cell (or facet) data.
    @process.register(ufl.classes.SpatialCoordinate)
    @process.register(ufl.classes.FacetJacobian)
    @process.register(ufl.classes.FacetJacobianDeterminant)
    @process.register(ufl.classes.FacetJacobianInverse)
    @process.register(ufl.classes.FacetArea)
    @process.register(ufl.classes.MinFacetEdgeLength)
    @process.register(ufl.classes.MaxFacetEdgeLength)
    @process.register(ufl.classes.FacetOrigin)
    def _(self, o):
        return self._default_restricted(o)

    @process.register(ufl.classes.Interpolate)
    def _(self, o: ufl.classes.Interpolate):
        """Restrict an interpolated finite element field."""
        if o.ufl_element() in H1:
            return self._default_restricted(o)
        else:
            return self._require_restriction(o)

    @process.register(ufl.classes.Coefficient)
    def _(self, o: ufl.classes.Coefficient):
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

    @process.register(ufl.classes.FacetNormal)
    def _(self, o: ufl.classes.FacetNormal):
        """Restrict a facet_normal."""
        D = cast(Mesh, extract_unique_domain(o))
        e = D.ufl_coordinate_element()
        gd = D.geometric_dimension
        td = D.topological_dimension

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


def apply_restrictions(
    expression: ufl.classes.Expr | Integral, default_restrictions: dict | None = None
) -> ufl.classes.Expr:
    """Propagate restriction nodes to wrap differential terminals directly.

    Args:
        expression:
            UFL expression.
        default_restrictions:
            domain-default_restriction map.
            If ``None``, just propagate restrictions without
            applying the default restrictions.

    Returns:
        expression with the restriction nodes propagated.

    """
    rules = RestrictionPropagator(default_restrictions=default_restrictions)
    return map_integrands(rules, expression)
