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

from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import (
    Argument,
    Coefficient,
    Constant,
    ConstantValue,
    Expr,
    FacetArea,
    FacetCoordinate,
    FacetJacobian,
    FacetJacobianDeterminant,
    FacetJacobianInverse,
    FacetNormal,
    FacetOrigin,
    GeometricCellQuantity,
    GeometricFacetQuantity,
    Grad,
    Interpolate,
    Label,
    MaxFacetEdgeLength,
    MinFacetEdgeLength,
    MultiIndex,
    Operator,
    QuadratureWeight,
    ReferenceCellVolume,
    ReferenceFacetVolume,
    ReferenceValue,
    Restricted,
    SpatialCoordinate,
    Terminal,
    Variable,
)
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
    def process(self, o: Expr):
        """Process an expression."""
        return super().process(o)

    @process.register(Operator)
    @DAGTraverser.postorder
    def _(self, o: Operator, *operands):
        """Reconstruct an operator if restriction propagation changed an operand."""
        if all(new is old for new, old in zip(operands, o.ufl_operands)):
            return o
        return o._ufl_expr_reconstruct_(*operands)

    @process.register(Restricted)
    def _(self, o: Restricted):
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
    @process.register(Grad)
    def _(self, o: Grad):
        return self._require_restriction(o)

    @process.register(Variable)
    @DAGTraverser.postorder
    def _(self, o: Variable, op, label):
        """Strip variable."""
        return op

    @process.register(ReferenceValue)
    def _(self, o: ReferenceValue):
        """Reference value of something follows same restriction rule as the underlying object."""
        (f,) = o.ufl_operands
        assert f._ufl_is_terminal_ or isinstance(f, Interpolate)
        g = self(f)
        if isinstance(g, Restricted):
            side = g.side()
            return o(side)
        else:
            return o

    # --- Rules for terminals

    # Require handlers to be specified for all terminals
    @process.register(Terminal)
    def _(self, o: Terminal):
        return self._missing_rule(o)

    @process.register(MultiIndex)
    def _(self, o: MultiIndex):
        return self._ignore_restriction(o)

    @process.register(Label)
    def _(self, o: Label):
        return self._ignore_restriction(o)

    # Default: Literals should ignore restriction
    @process.register(ConstantValue)
    def _(self, o: ConstantValue):
        return self._ignore_restriction(o)

    @process.register(Constant)
    def _(self, o: Constant):
        return self._ignore_restriction(o)

    # Even arguments with continuous elements such as Lagrange must be
    # restricted to associate with the right part of the element
    # matrix
    @process.register(Argument)
    def _(self, o: Argument):
        return self._require_restriction(o)

    # Defaults for geometric quantities
    @process.register(GeometricCellQuantity)
    def _(self, o: GeometricCellQuantity):
        return self._require_restriction(o)

    @process.register(GeometricFacetQuantity)
    def _(self, o: GeometricFacetQuantity):
        return self._require_restriction(o)

    # Only a few geometric quantities are independent on the restriction:
    @process.register(FacetCoordinate)
    def _(self, o: FacetCoordinate):
        return self._ignore_restriction(o)

    @process.register(QuadratureWeight)
    def _(self, o: QuadratureWeight):
        return self._ignore_restriction(o)

    # Assuming homogeoneous mesh
    @process.register(ReferenceCellVolume)
    def _(self, o: ReferenceCellVolume):
        return self._ignore_restriction(o)

    @process.register(ReferenceFacetVolume)
    def _(self, o: ReferenceFacetVolume):
        return self._ignore_restriction(o)

    # These are the same from either side but to compute them
    # cell (or facet) data from one side must be selected:
    @process.register(SpatialCoordinate)
    def _(self, o: SpatialCoordinate):
        return self._default_restricted(o)

    # Depends on cell only to get to the facet:
    @process.register(FacetJacobian)
    def _(self, o: FacetJacobian):
        return self._default_restricted(o)

    @process.register(FacetJacobianDeterminant)
    def _(self, o: FacetJacobianDeterminant):
        return self._default_restricted(o)

    @process.register(FacetJacobianInverse)
    def _(self, o: FacetJacobianInverse):
        return self._default_restricted(o)

    @process.register(FacetArea)
    def _(self, o: FacetArea):
        return self._default_restricted(o)

    @process.register(MinFacetEdgeLength)
    def _(self, o: MinFacetEdgeLength):
        return self._default_restricted(o)

    @process.register(MaxFacetEdgeLength)
    def _(self, o: MaxFacetEdgeLength):
        return self._default_restricted(o)

    @process.register(FacetOrigin)
    def _(self, o: FacetOrigin):
        return self._default_restricted(o)

    @process.register(Interpolate)
    def _(self, o: Interpolate):
        """Restrict an interpolated finite element field."""
        if o.ufl_element() in H1:
            return self._default_restricted(o)
        else:
            return self._require_restriction(o)

    @process.register(Coefficient)
    def _(self, o: Coefficient):
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

    @process.register(FacetNormal)
    def _(self, o: FacetNormal):
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
    expression: Expr | Integral, default_restrictions: dict | None = None
) -> Expr:
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
