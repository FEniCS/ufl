"Algorithms related to restrictions."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from ufl.expr import Expr
from ufl.classes import Measure
from ufl.assertions import ufl_assert
from ufl.algorithms.transformer import Transformer, ReuseTransformer, apply_transformer

class RestrictionPropagator(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)
        self.current_restriction = None
        self.default_restriction = "+"
        self._variables = {}

    def restricted(self, o):
        "When hitting a restricted quantity, store the restriction state, visit child, and reset the restriction state."
        prev_restricted = self.current_restriction

        ufl_assert(prev_restricted is None or prev_restricted == o._side,
                   "Cannot restrict to different sides.")

        self.current_restriction = o._side
        e, = o.operands()
        r = self.visit(e)

        self.current_restriction = prev_restricted
        return r

    def _ignore_restriction(self, o):
        "Ignore current restriction, quantity is independent of side also from a computational point of view."
        return o

    def _require_restriction(self, o):
        "Restrict a discontinuous quantity to current side, require a side to be set."
        ufl_assert(self.current_restriction is not None, "Discontinuous type %s must be restricted." % o._uflclass.__name__)
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
            ufl_error("Discontinuous type %s must be restricted." % o._uflclass.__name__)
        elif self.current_restriction == self.default_restriction:
            return o(self.default_restriction)
        else:
            return -o(self.default_restriction)

    def _missing_rule(self, o):
        error("Missing rule for %s" % o._uflclass.__name__)

    # Default: Literals should ignore restriction
    terminal = _ignore_restriction

    # Even arguments with continuous elements such as Lagrange must be
    # restricted to associate with the right part of the element matrix
    argument = _require_restriction

    def coefficient(self, o):
        "Allow coefficients to be unrestricted (apply default if so) if the values are fully continuous across the facet."
        e = o.element()
        d = e.degree()
        f = e.family()
        # TODO: Move this choice to the element class?
        if (f == "Lagrange" and d > 0) or f == "Real":
            # If the coefficient _value_ is _fully_ continuous
            return self._default_restricted(o) # Must still be computed from one of the sides, don't care which
        else:
            return self._require_restriction(o)

    # Defaults for geometric quantities
    geometric_cell_quantity = _missing_rule  #_require_restriction
    geometric_facet_quantity = _missing_rule #_ignore_restriction

    spatial_coordinate = _default_restricted # Continuous but computed from cell data
    cell_coordinate = _require_restriction   # Depends on cell
    facet_coordinate = _ignore_restriction   # Independent of cell

    cell_origo = _require_restriction       # Depends on cell
    facet_origo = _default_restricted       # Depends on cell but only to get to the facet # TODO: Is this valid for quads?
    cell_facet_origo = _require_restriction # Depends on cell

    jacobian = _require_restriction             # Property of cell
    jacobian_determinant = _require_restriction # ...
    jacobian_inverse = _require_restriction     # ...

    facet_jacobian = _default_restricted              # Depends on cell only to get to the facet
    facet_jacobian_determinant = _default_restricted  # ... (actually continuous?)
    facet_jacobian_inverse = _default_restricted      # ...

    cell_facet_jacobian = _require_restriction             # Depends on cell
    cell_facet_jacobian_determinant = _require_restriction # ...
    cell_facet_jacobian_inverse = _require_restriction     # ...

    #facet_normal = _opposite           # Opposite pointing vector depending on cell # Enabling this changes the FFC reference code too much, will test if the performance is just as good later.
    facet_normal = _require_restriction # Direction depends on cell, make it explicit
    cell_normal = _require_restriction  # Property of cell

    #facet_tangents = _default_restricted # Independent of cell
    #cell_tangents = _require_restriction # Depends on cell
    #cell_midpoint = _require_restriction # Depends on cell
    #facet_midpoint = _default_restricted # Depends on cell only to get to the facet

    cell_volume = _require_restriction        # Property of cell
    circumradius = _require_restriction       # Property of cell
    #cell_surface_area = _require_restriction # Property of cell

    facet_area = _default_restricted            # Depends on cell only to get to the facet
    #facet_diameter = _default_restricted       # Depends on cell only to get to the facet
    min_facet_edge_length = _default_restricted # Depends on cell only to get to the facet
    max_facet_edge_length = _default_restricted # Depends on cell only to get to the facet

    cell_orientation = _require_restriction # Property of cell
    quadrature_weight = _ignore_restriction # Independent of cell

    def variable(self, o, e, l):
        "Recreate variable with same label but cache so it only happens once."
        v = self._variables.get(l)
        if v is None:
            v = self.visit(e)
            self._variables[l] = v
        return v

class RestrictionChecker(Transformer):
    def __init__(self, require_restriction):
        Transformer.__init__(self)
        self.current_restriction = None
        self.require_restriction = require_restriction

    def expr(self, o):
        pass

    def restricted(self, o):
        ufl_assert(self.current_restriction is None,
            "Not expecting twice restricted expression.")
        self.current_restriction = o._side
        e, = o.operands()
        self.visit(e)
        self.current_restriction = None

    def facet_normal(self, o):
        if self.require_restriction:
            ufl_assert(self.current_restriction is not None, "Facet normal must be restricted in interior facet integrals.")
        else:
            ufl_assert(self.current_restriction is None, "Restrictions are only allowed for interior facet integrals.")

    def form_argument(self, o):
        if self.require_restriction:
            ufl_assert(self.current_restriction is not None, "Form argument must be restricted in interior facet integrals.")
        else:
            ufl_assert(self.current_restriction is None, "Restrictions are only allowed for interior facet integrals.")

def propagate_restrictions(expression):
    "Propagate restriction nodes to wrap terminal objects directly."
    return apply_transformer(expression, RestrictionPropagator(), integral_type=Measure.INTERIOR_FACET)

def check_restrictions(expression, require_restriction):
    ufl_assert(isinstance(expression, Expr), "Expecting Expr instance.")
    return RestrictionChecker(require_restriction).visit(expression)
