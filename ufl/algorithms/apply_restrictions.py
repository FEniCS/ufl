# -*- coding: utf-8 -*-
"""This module contains the apply_restrictions algorithm which propagates restrictions in a form towards the terminals."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.


from ufl.log import error
from ufl.classes import Restricted
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.measure import integral_type_to_measure_name


class RestrictionPropagator(MultiFunction):
    def __init__(self, side=None):
        MultiFunction.__init__(self)
        self.current_restriction = side
        self.default_restriction = "+"
        if self.current_restriction is None:
            self._rp = {"+": RestrictionPropagator("+"),
                        "-": RestrictionPropagator("-")}

    def restricted(self, o):
        "When hitting a restricted quantity, visit child with a separate restriction algorithm."
        # Assure that we have only two levels here, inside or outside
        # the Restricted node
        if self.current_restriction is not None:
            error("Cannot restrict an expression twice.")
        # Configure a propagator for this side and apply to subtree
        return map_expr_dag(self._rp[o.side()], o.ufl_operands[0])  # FIXME: Reuse cache between these calls!

    # --- Reusable rules

    def _ignore_restriction(self, o):
        "Ignore current restriction, quantity is independent of side also from a computational point of view."
        return o

    def _require_restriction(self, o):
        "Restrict a discontinuous quantity to current side, require a side to be set."
        if self.current_restriction is None:
            error("Discontinuous type %s must be restricted." % o._ufl_class_.__name__)
        return o(self.current_restriction)

    def _default_restricted(self, o):
        "Restrict a continuous quantity to default side if no current restriction is set."
        r = self.current_restriction
        if r is None:
            r = self.default_restriction
        return o(r)

    def _opposite(self, o):
        "Restrict a quantity to default side, if the current restriction is different swap the sign, require a side to be set."
        if self.current_restriction is None:
            error("Discontinuous type %s must be restricted." % o._ufl_class_.__name__)
        elif self.current_restriction == self.default_restriction:
            return o(self.default_restriction)
        else:
            return -o(self.default_restriction)

    def _missing_rule(self, o):
        error("Missing rule for %s" % o._ufl_class_.__name__)

    # --- Rules for operators

    # Default: Operators should reconstruct only if subtrees are not
    # touched
    operator = MultiFunction.reuse_if_untouched

    # Assuming apply_derivatives has been called, propagating Grad
    # inside the Restricted nodes.
    grad = _require_restriction  # Considering all grads to be discontinuous, may need something else for facet functions in future
    # Assuming averages are also applied directly to the terminal or grad nodes
    cell_avg = _require_restriction
    facet_avg = _ignore_restriction

    def variable(self, o, op, label):
        "Strip variable."
        return op

    def reference_value(self, o):
        "Reference value of something follows same restriction rule as the underlying object."
        f, = o.ufl_operands
        assert f._ufl_is_terminal_
        g = self(f)
        if isinstance(g, Restricted):
            side = g.side()
            return o(side)
        else:
            return o

    # --- Rules for terminals

    # Default: Literals should ignore restriction
    terminal = _ignore_restriction  # TODO: Require handlers to be specified for all terminals? That would be safer.

    # Even arguments with continuous elements such as Lagrange must be
    # restricted to associate with the right part of the element
    # matrix
    argument = _require_restriction

    def coefficient(self, o):
        "Allow coefficients to be unrestricted (apply default if so) if the values are fully continuous across the facet."
        e = o.ufl_element()
        d = e.degree()
        f = e.family()
        # TODO: Move this choice to the element class?
        if (f == "Lagrange" and d > 0) or f == "Real":
            # If the coefficient _value_ is _fully_ continuous
            return self._default_restricted(o)  # Must still be computed from one of the sides, we just don't care which
        else:
            return self._require_restriction(o)

    def facet_normal(self, o):
        D = o.ufl_domain()
        e = D.ufl_coordinate_element()
        f = e.family()
        d = e.degree()
        gd = D.geometric_dimension()
        td = D.topological_dimension()

        if f == "Lagrange" and d == 1 and gd == td:
            # For meshes with a continuous linear non-manifold
            # coordinate field, the facet normal from side - points in
            # the opposite direction of the one from side +.  We must
            # still require a side to be chosen by the user but
            # rewrite n- -> n+.  This is an optimization, possibly
            # premature, however it's more difficult to do at a later
            # stage.
            return self._opposite(o)
        else:
            # For other meshes, we require a side to be chosen by the
            # user and respect that
            return self._require_restriction(o)

    # Although the physical normal can be flipped when moving from +
    # to -, the reference normal cannot
    reference_normal = _require_restriction

    # Defaults for geometric quantities
    geometric_cell_quantity = _missing_rule  # _require_restriction
    geometric_facet_quantity = _missing_rule  # _ignore_restriction

    spatial_coordinate = _default_restricted  # Continuous but computed from cell data
    cell_coordinate = _require_restriction  # Depends on cell
    facet_coordinate = _ignore_restriction  # Independent of cell

    cell_origin = _require_restriction        # Depends on cell
    facet_origin = _default_restricted        # Depends on cell but only to get to the facet # TODO: Is this valid for quads?
    cell_facet_origin = _require_restriction  # Depends on cell

    jacobian = _require_restriction              # Property of cell
    jacobian_determinant = _require_restriction  # ...
    jacobian_inverse = _require_restriction      # ...

    facet_jacobian = _default_restricted              # Depends on cell only to get to the facet
    facet_jacobian_determinant = _default_restricted  # ... (actually continuous?)
    facet_jacobian_inverse = _default_restricted      # ...

    cell_facet_jacobian = _require_restriction              # Depends on cell
    cell_facet_jacobian_determinant = _require_restriction  # ...
    cell_facet_jacobian_inverse = _require_restriction      # ...
    cell_edge_vectors = _require_restriction                # ...

    reference_cell_volume = _ignore_restriction   # FIXME: needs changing for mixed cell meshes
    reference_facet_volume = _ignore_restriction  # FIXME: needs changing for mixed cell meshes

    cell_normal = _require_restriction  # Property of cell

    # facet_tangents = _default_restricted # Independent of cell
    # cell_tangents = _require_restriction # Depends on cell
    # cell_midpoint = _require_restriction # Depends on cell
    # facet_midpoint = _default_restricted # Depends on cell only to get to the facet

    cell_volume = _require_restriction         # Property of cell
    circumradius = _require_restriction        # Property of cell
    # cell_surface_area = _require_restriction  # Property of cell

    facet_area = _default_restricted             # Depends on cell only to get to the facet
    # facet_diameter = _default_restricted       # Depends on cell only to get to the facet
    min_facet_edge_length = _default_restricted  # Depends on cell only to get to the facet
    max_facet_edge_length = _default_restricted  # Depends on cell only to get to the facet

    cell_orientation = _require_restriction   # Property of cell
    facet_orientation = _require_restriction  # Property of cell (depends on local facet number in cell)
    quadrature_weight = _ignore_restriction   # Independent of cell


def apply_restrictions(expression):
    "Propagate restriction nodes to wrap differential terminals directly."
    integral_types = [k for k in integral_type_to_measure_name.keys()
                      if k.startswith("interior_facet")]
    rules = RestrictionPropagator()
    return map_integrand_dags(rules, expression,
                              only_integral_type=integral_types)
